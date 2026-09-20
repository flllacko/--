from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.config import settings
from app.db.init_db import initialize_database, utc_now

from .processor import ProcessingResult, process_markdown_file


VersionRelation = Literal["旧资料新版本", "独立资料"]


class VersionError(ValueError):
    """版本关系选择或资料关系不合法。"""


@dataclass(frozen=True)
class VersionResult:
    relation: VersionRelation
    status: Literal["处理成功", "处理失败"]
    old_document_id: int | None
    new_document_id: int | None
    duplicate: bool = False
    error_message: str | None = None


def process_with_version_choice(
    file_path: Path,
    *,
    relation: VersionRelation,
    old_document_id: int | None = None,
    source_root: Path | None = None,
    db_path: Path | None = None,
    index_path: Path | None = None,
    ollama_base_url: str | None = None,
    embedding_model: str | None = None,
) -> VersionResult:
    """按管理员选择处理独立资料或旧资料的新版本。"""

    root = (source_root or settings.public_source_root).resolve()
    database = (db_path or settings.sqlite_path).resolve()
    vector_index = (index_path or settings.lancedb_path).resolve()
    initialize_database(database)

    if relation == "独立资料":
        result = process_markdown_file(
            file_path,
            source_root=root,
            db_path=database,
            index_path=vector_index,
            ollama_base_url=ollama_base_url,
            embedding_model=embedding_model,
        )
        return VersionResult(
            relation=relation,
            status=result.status,
            old_document_id=None,
            new_document_id=result.document_id,
            duplicate=result.duplicate,
            error_message=result.error_message,
        )

    if old_document_id is None:
        raise VersionError("选择“旧资料新版本”时必须指定旧资料")

    connection = sqlite3.connect(database)
    try:
        old = connection.execute(
            """
            SELECT stable_key, processing_status, validity_status, version_group_key
            FROM knowledge_documents
            WHERE id = ?
            """,
            (old_document_id,),
        ).fetchone()
    finally:
        connection.close()
    if not old:
        raise VersionError(f"找不到旧资料：{old_document_id}")
    if old[1] != "处理成功":
        raise VersionError("只有处理成功的资料才能作为旧版本")

    path = file_path if file_path.is_absolute() else root / file_path
    path = path.resolve()
    try:
        path.relative_to(root)
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    except ValueError as exc:
        raise VersionError("新版本资料必须位于公共资料正文目录内") from exc
    except OSError as exc:
        return VersionResult(
            relation=relation,
            status="处理失败",
            old_document_id=old_document_id,
            new_document_id=None,
            error_message=f"读取新版本资料失败：{exc}",
        )

    version_group_key = old[3] or old[0]
    stable_key = f"public:markdown-version:{old_document_id}:{content_hash}"
    result = process_markdown_file(
        path,
        source_root=root,
        db_path=database,
        index_path=vector_index,
        ollama_base_url=ollama_base_url,
        embedding_model=embedding_model,
        stable_key_override=stable_key,
        version_group_key=version_group_key,
        supersedes_document_id=old_document_id,
    )
    if result.status != "处理成功":
        return VersionResult(
            relation=relation,
            status="处理失败",
            old_document_id=old_document_id,
            new_document_id=result.document_id,
            error_message=result.error_message,
        )
    if result.duplicate or result.document_id == old_document_id:
        return VersionResult(
            relation=relation,
            status="处理成功",
            old_document_id=old_document_id,
            new_document_id=result.document_id,
            duplicate=True,
            error_message="新版本内容与已有成功资料完全相同，未切换旧版本",
        )

    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            connection.execute(
                """
                UPDATE knowledge_documents
                SET validity_status = '无效', updated_at = ?
                WHERE id = ? AND processing_status = '处理成功'
                """,
                (utc_now(), old_document_id),
            )
            connection.execute(
                """
                UPDATE knowledge_documents
                SET validity_status = '有效', version_group_key = ?,
                    supersedes_document_id = ?, updated_at = ?
                WHERE id = ? AND processing_status = '处理成功'
                """,
                (version_group_key, old_document_id, utc_now(), result.document_id),
            )
    finally:
        connection.close()

    return VersionResult(
        relation=relation,
        status="处理成功",
        old_document_id=old_document_id,
        new_document_id=result.document_id,
    )

