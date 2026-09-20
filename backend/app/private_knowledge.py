from __future__ import annotations

"""普通用户私人资料的所有权、文件路径和删除边界。

这个模块只返回私人资料元信息，不提供正文或切片读取接口。正文只在检索层
按 owner_user_id 过滤后进入当前账号自己的回答流程。
"""

import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.auth import connect_database
from app.processing.lance_store import LanceStoreError, remove_document


class PrivateKnowledgeError(ValueError):
    pass


@dataclass(frozen=True)
class PrivateDeletionPlan:
    document_id: int
    title: str
    source_path: Path | None
    confirmation_token: str


@dataclass(frozen=True)
class PrivateDeletionResult:
    document_id: int
    deleted_source_file: bool
    deleted_vector_rows: bool


class PrivateKnowledgeService:
    def __init__(self, db_path: Path, source_root: Path, index_path: Path) -> None:
        self.db_path = db_path.resolve()
        self.source_root = source_root.resolve()
        self.index_path = index_path.resolve()
        self._plans: dict[str, PrivateDeletionPlan] = {}

    def list_documents(self, user_id: int, query: str = "", limit: int = 100) -> list[dict[str, Any]]:
        clauses = [
            "d.owner_user_id = ?",
            "d.is_private = 1",
            "d.processing_status = '处理成功'",
        ]
        params: list[object] = [user_id]
        if query.strip():
            clauses.append("(d.title LIKE ? OR d.source_label LIKE ?)")
            pattern = f"%{query.strip()}%"
            params.extend([pattern, pattern])
        where = " AND ".join(clauses)
        connection = connect_database(self.db_path)
        try:
            rows = connection.execute(
                f"""
                SELECT d.id, d.title, d.source_label, d.processing_status,
                       d.created_at, d.updated_at
                FROM knowledge_documents d
                WHERE {where}
                ORDER BY d.created_at DESC, d.id DESC
                LIMIT ?
                """,
                [*params, max(1, min(limit, 200))],
            ).fetchall()
            return [
                {
                    "id": int(row["id"]),
                    "title": str(row["title"]),
                    "filename": str(row["source_label"] or ""),
                    "processing_status": "处理成功",
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in rows
            ]
        finally:
            connection.close()

    def has_successful_documents(self, user_id: int) -> bool:
        connection = connect_database(self.db_path)
        try:
            row = connection.execute(
                """
                SELECT 1 FROM knowledge_documents
                WHERE owner_user_id = ? AND is_private = 1
                  AND processing_status = '处理成功'
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            return row is not None
        finally:
            connection.close()

    def processing_result(self, user_id: int, document_id: int) -> dict[str, Any]:
        connection = connect_database(self.db_path)
        try:
            row = connection.execute(
                """
                SELECT d.id, d.title, d.source_label, d.processing_status,
                       (SELECT t.error_message FROM processing_tasks t
                        WHERE t.document_id = d.id ORDER BY t.id DESC LIMIT 1) AS error_message
                FROM knowledge_documents d
                WHERE d.id = ? AND d.owner_user_id = ? AND d.is_private = 1
                """,
                (document_id, user_id),
            ).fetchone()
            if row is None:
                raise PrivateKnowledgeError("找不到属于当前账号的私人资料。")
            status = str(row["processing_status"])
            return {
                "document_id": int(row["id"]),
                "title": str(row["title"]),
                "filename": str(row["source_label"] or ""),
                "status": "处理成功" if status == "处理成功" else "处理失败",
                "error_message": row["error_message"],
            }
        finally:
            connection.close()

    def update_title(self, user_id: int, document_id: int, title: str) -> dict[str, Any]:
        cleaned = " ".join(title.strip().split())
        if not cleaned:
            raise PrivateKnowledgeError("标题不能为空。")
        if len(cleaned) > 120:
            raise PrivateKnowledgeError("标题不能超过 120 个字符。")
        connection = connect_database(self.db_path)
        try:
            with connection:
                updated = connection.execute(
                    """
                    UPDATE knowledge_documents
                    SET title = ?, updated_at = datetime('now')
                    WHERE id = ? AND owner_user_id = ? AND is_private = 1
                    """,
                    (cleaned, document_id, user_id),
                ).rowcount
            if updated != 1:
                raise PrivateKnowledgeError("找不到属于当前账号的私人资料。")
            return {"document_id": document_id, "title": cleaned}
        finally:
            connection.close()

    def source_path(self, user_id: int, document_id: int) -> Path:
        connection = connect_database(self.db_path)
        try:
            row = connection.execute(
                """
                SELECT stable_key, source_label
                FROM knowledge_documents
                WHERE id = ? AND owner_user_id = ? AND is_private = 1
                """,
                (document_id, user_id),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise PrivateKnowledgeError("找不到属于当前账号的私人资料。")

        stable_key = str(row["stable_key"])
        parts = stable_key.split(":", 3)
        if len(parts) != 4 or parts[0] != "private" or parts[2] != "file":
            raise PrivateKnowledgeError("该私人资料没有可用的原始文件路径。")
        try:
            stable_owner = int(parts[1])
        except ValueError as exc:
            raise PrivateKnowledgeError("私人资料所属账号标识无效。") from exc
        if stable_owner != user_id:
            raise PrivateKnowledgeError("不能访问其他账号的私人资料。")
        user_root = (self.source_root / str(user_id)).resolve()
        candidate = (user_root / Path(parts[3])).resolve()
        try:
            candidate.relative_to(user_root)
        except ValueError as exc:
            raise PrivateKnowledgeError("原始文件路径越过私人资料目录，拒绝访问。") from exc
        return candidate

    def create_delete_plan(self, user_id: int, document_id: int) -> PrivateDeletionPlan:
        connection = connect_database(self.db_path)
        try:
            row = connection.execute(
                """
                SELECT id, title
                FROM knowledge_documents
                WHERE id = ? AND owner_user_id = ? AND is_private = 1
                """,
                (document_id, user_id),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise PrivateKnowledgeError("找不到属于当前账号的私人资料。")
        try:
            source_path = self.source_path(user_id, document_id)
        except PrivateKnowledgeError:
            source_path = None
        plan = PrivateDeletionPlan(
            document_id=int(row["id"]),
            title=str(row["title"]),
            source_path=source_path,
            confirmation_token=secrets.token_urlsafe(24),
        )
        self._plans[plan.confirmation_token] = plan
        return plan

    def confirm_and_delete(self, user_id: int, confirmation_token: str) -> PrivateDeletionResult:
        plan = self._plans.pop(confirmation_token, None)
        if plan is None:
            raise PrivateKnowledgeError("删除确认已失效，请重新生成删除计划。")

        connection = connect_database(self.db_path)
        try:
            owner = connection.execute(
                """
                SELECT owner_user_id, is_private
                FROM knowledge_documents WHERE id = ?
                """,
                (plan.document_id,),
            ).fetchone()
            if owner is None or int(owner["owner_user_id"] or -1) != user_id or int(owner["is_private"]) != 1:
                raise PrivateKnowledgeError("不能删除不属于当前账号的私人资料。")
        finally:
            connection.close()

        try:
            remove_document(self.index_path, plan.document_id)
            vector_deleted = True
        except LanceStoreError as exc:
            self._plans[confirmation_token] = plan
            raise PrivateKnowledgeError(f"删除本地检索副本失败，未删除私人资料记录：{exc}") from exc

        connection = connect_database(self.db_path)
        try:
            with connection:
                deleted = connection.execute(
                    "DELETE FROM knowledge_documents WHERE id = ? AND owner_user_id = ? AND is_private = 1",
                    (plan.document_id, user_id),
                ).rowcount
            if deleted != 1:
                raise PrivateKnowledgeError("私人资料在确认后已经不存在。")
        finally:
            connection.close()

        deleted_source = False
        if plan.source_path and plan.source_path.exists():
            try:
                plan.source_path.unlink()
                deleted_source = True
            except OSError:
                # 业务记录和检索副本已经删除，不能因为原文件异常而回滚数据库。
                deleted_source = False
        return PrivateDeletionResult(
            document_id=plan.document_id,
            deleted_source_file=deleted_source,
            deleted_vector_rows=vector_deleted,
        )

    def source_paths_for_user(self, user_id: int) -> list[Path]:
        connection = connect_database(self.db_path)
        try:
            rows = connection.execute(
                """
                SELECT stable_key FROM knowledge_documents
                WHERE owner_user_id = ? AND is_private = 1
                """,
                (user_id,),
            ).fetchall()
        finally:
            connection.close()
        paths: list[Path] = []
        for row in rows:
            try:
                paths.append(self._source_path_from_key(user_id, str(row["stable_key"])))
            except PrivateKnowledgeError:
                continue
        return paths

    def document_ids_for_user(self, user_id: int) -> list[int]:
        connection = connect_database(self.db_path)
        try:
            rows = connection.execute(
                """
                SELECT id FROM knowledge_documents
                WHERE owner_user_id = ? AND is_private = 1
                ORDER BY id
                """,
                (user_id,),
            ).fetchall()
            return [int(row["id"]) for row in rows]
        finally:
            connection.close()

    def _source_path_from_key(self, user_id: int, stable_key: str) -> Path:
        parts = stable_key.split(":", 3)
        if len(parts) != 4 or parts[0] != "private" or parts[2] != "file":
            raise PrivateKnowledgeError("私人资料路径无效。")
        try:
            stable_owner = int(parts[1])
        except ValueError as exc:
            raise PrivateKnowledgeError("私人资料路径所属账号无效。") from exc
        if stable_owner != user_id:
            raise PrivateKnowledgeError("私人资料路径所属账号不匹配。")
        user_root = (self.source_root / str(user_id)).resolve()
        candidate = (user_root / Path(parts[3])).resolve()
        try:
            candidate.relative_to(user_root)
        except ValueError as exc:
            raise PrivateKnowledgeError("私人资料路径越界。") from exc
        return candidate


def remove_private_source_files(service: PrivateKnowledgeService, user_id: int) -> None:
    for path in service.source_paths_for_user(user_id):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue
