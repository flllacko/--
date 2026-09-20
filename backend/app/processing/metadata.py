from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from app.db.init_db import initialize_database, utc_now

from .markdown import normalize_content_status


ALLOWED_CONTENT_STATUS = {"可直接回答", "历史可回答", "只能提供入口", "禁止作为依据"}
ALLOWED_VALIDITY_STATUS = {"有效", "无效"}


class MetadataError(ValueError):
    """资料元信息不符合项目约束。"""


@dataclass(frozen=True)
class DocumentMetadata:
    document_id: int
    title: str
    module: str
    source_label: str
    source_url: str | None
    published_date: str | None
    publisher_department: str | None
    content_status: str
    validity_status: str


def update_document_metadata(
    db_path: Path,
    document_id: int,
    *,
    title: str | None = None,
    module: str | None = None,
    source_label: str | None = None,
    source_url: str | None = None,
    published_date: str | None = None,
    publisher_department: str | None = None,
    content_status: str | None = None,
    validity_status: str | None = None,
) -> DocumentMetadata:
    """更新可编辑元信息，不重新切片或生成向量。"""

    initialize_database(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        row = connection.execute(
            """
            SELECT title, module, source_label, source_url, published_date,
                   publisher_department, content_status, validity_status
            FROM knowledge_documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
        if not row:
            raise MetadataError(f"找不到资料：{document_id}")

        next_title = (title if title is not None else row[0]).strip()
        next_module = (module if module is not None else row[1]).strip()
        next_source_label = (source_label if source_label is not None else row[2]).strip()
        next_source_url = source_url.strip() if source_url is not None and source_url.strip() else None
        next_published_date = (
            published_date.strip() if published_date is not None and published_date.strip() else None
        )
        next_department = (
            publisher_department.strip()
            if publisher_department is not None and publisher_department.strip()
            else None
        )
        next_content_status = normalize_content_status(
            content_status if content_status is not None else row[6]
        )
        next_validity_status = validity_status if validity_status is not None else row[7]

        if not next_title:
            raise MetadataError("标题不能为空")
        if not next_module:
            raise MetadataError("所属模块不能为空")
        if not next_source_label:
            raise MetadataError("来源文字不能为空")
        if next_source_url and urlparse(next_source_url).scheme not in {"http", "https"}:
            raise MetadataError("原文链接只能使用 http 或 https，或留空")
        if next_content_status not in ALLOWED_CONTENT_STATUS:
            raise MetadataError("内容状态不在项目规定的四种值中")
        if next_validity_status not in ALLOWED_VALIDITY_STATUS:
            raise MetadataError("时效状态只能是有效或无效")

        now = utc_now()
        with connection:
            connection.execute(
                """
                UPDATE knowledge_documents
                SET title = ?, module = ?, source_label = ?, source_url = ?,
                    published_date = ?, publisher_department = ?,
                    content_status = ?, validity_status = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    next_title,
                    next_module,
                    next_source_label,
                    next_source_url,
                    next_published_date,
                    next_department,
                    next_content_status,
                    next_validity_status,
                    now,
                    document_id,
                ),
            )
            source = connection.execute(
                "SELECT id FROM source_records WHERE document_id = ? ORDER BY id LIMIT 1",
                (document_id,),
            ).fetchone()
            if source:
                connection.execute(
                    """
                    UPDATE source_records
                    SET label = ?, url = ?, created_at = ?
                    WHERE id = ?
                    """,
                    (next_source_label, next_source_url, now, source[0]),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO source_records (document_id, label, url, locator, created_at)
                    VALUES (?, ?, ?, '元信息编辑', ?)
                    """,
                    (document_id, next_source_label, next_source_url, now),
                )

        return DocumentMetadata(
            document_id=document_id,
            title=next_title,
            module=next_module,
            source_label=next_source_label,
            source_url=next_source_url,
            published_date=next_published_date,
            publisher_department=next_department,
            content_status=next_content_status,
            validity_status=next_validity_status,
        )
    finally:
        connection.close()
