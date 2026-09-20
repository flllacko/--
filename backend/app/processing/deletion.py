from __future__ import annotations

import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.config import settings
from app.db.init_db import initialize_database, utc_now

from .lance_store import LanceStoreError, remove_document


class DeletionError(ValueError):
    """资料删除前置检查或确认失败。"""


@dataclass(frozen=True)
class DeletionPlan:
    document_id: int
    title: str
    source_path: Path | None
    chunk_count: int
    source_record_count: int
    task_count: int
    confirmation_token: str


@dataclass(frozen=True)
class DeletionResult:
    document_id: int
    deleted_source_file: bool
    deleted_vector_rows: bool


class DeletionService:
    """通过“生成计划 → 第二次确认”删除公共资料及其检索副本。"""

    def __init__(
        self,
        db_path: Path | None = None,
        index_path: Path | None = None,
        source_root: Path | None = None,
    ) -> None:
        self.db_path = (db_path or settings.sqlite_path).resolve()
        self.index_path = (index_path or settings.lancedb_path).resolve()
        self.source_root = (source_root or settings.public_source_root).resolve()
        self._plans: dict[str, DeletionPlan] = {}
        initialize_database(self.db_path)

    def create_plan(self, document_id: int, source_path: Path | None = None) -> DeletionPlan:
        connection = sqlite3.connect(self.db_path)
        try:
            row = connection.execute(
                "SELECT title, stable_key FROM knowledge_documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            if not row:
                raise DeletionError(f"找不到资料：{document_id}")
            chunk_count = connection.execute(
                "SELECT COUNT(*) FROM knowledge_chunks WHERE document_id = ?",
                (document_id,),
            ).fetchone()[0]
            source_record_count = connection.execute(
                "SELECT COUNT(*) FROM source_records WHERE document_id = ?",
                (document_id,),
            ).fetchone()[0]
            task_count = connection.execute(
                "SELECT COUNT(*) FROM processing_tasks WHERE document_id = ?",
                (document_id,),
            ).fetchone()[0]
        finally:
            connection.close()

        resolved_source = self._resolve_source_path(row[1], source_path)
        token = secrets.token_urlsafe(24)
        plan = DeletionPlan(
            document_id=int(document_id),
            title=row[0],
            source_path=resolved_source,
            chunk_count=int(chunk_count),
            source_record_count=int(source_record_count),
            task_count=int(task_count),
            confirmation_token=token,
        )
        self._plans[token] = plan
        return plan

    def confirm_and_delete(self, confirmation_token: str) -> DeletionResult:
        plan = self._plans.pop(confirmation_token, None)
        if plan is None:
            raise DeletionError("删除确认已失效，请重新生成删除计划")

        try:
            remove_document(self.index_path, plan.document_id)
            vector_deleted = True
        except LanceStoreError as exc:
            self._plans[confirmation_token] = plan
            raise DeletionError(f"删除 LanceDB 检索副本失败，未删除业务记录：{exc}") from exc

        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                deleted = connection.execute(
                    "DELETE FROM knowledge_documents WHERE id = ?",
                    (plan.document_id,),
                ).rowcount
                if deleted != 1:
                    raise DeletionError(f"资料在确认后已经不存在：{plan.document_id}")
        finally:
            connection.close()

        deleted_source = False
        if plan.source_path and plan.source_path.exists():
            try:
                plan.source_path.unlink()
                deleted_source = True
            except OSError as exc:
                raise DeletionError(
                    f"业务记录和向量已删除，但原始文件删除失败，请按计划路径重试：{exc}"
                ) from exc

        return DeletionResult(
            document_id=plan.document_id,
            deleted_source_file=deleted_source,
            deleted_vector_rows=vector_deleted,
        )

    def _resolve_source_path(self, stable_key: str, explicit_path: Path | None) -> Path | None:
        candidate = explicit_path
        if candidate is None and stable_key.startswith("public:markdown:"):
            relative = stable_key.removeprefix("public:markdown:")
            candidate = self.source_root / Path(relative)
        if candidate is None:
            return None
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.source_root)
        except ValueError as exc:
            raise DeletionError("原始文件路径不在公共资料根目录内，拒绝删除") from exc
        return resolved

