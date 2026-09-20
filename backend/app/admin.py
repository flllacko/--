from __future__ import annotations

"""P8 管理后台的只读/受控管理服务。

管理员接口只返回公共资料元信息和普通用户账号状态，不返回学生问答正文、
密码哈希或私人资料正文。操作记录也只写入管理动作，不写入学生问题和回答。
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.auth import DEFAULT_ADMIN_PASSWORD, connect_database, hash_password


class AdminError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cleanup_old_logs(connection: sqlite3.Connection) -> None:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    connection.execute("DELETE FROM operation_logs WHERE created_at < ?", (cutoff,))


def record_operation(
    connection: sqlite3.Connection,
    *,
    actor_user_id: int | None,
    action: str,
    target_type: str,
    target_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    _cleanup_old_logs(connection)
    connection.execute(
        """
        INSERT INTO operation_logs
            (actor_user_id, action, target_type, target_id, detail_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            actor_user_id,
            action,
            target_type,
            target_id,
            json.dumps(detail or {}, ensure_ascii=False),
            _now(),
        ),
    )


def list_users(db_path: Path | str) -> list[dict[str, Any]]:
    connection = connect_database(db_path)
    try:
        rows = connection.execute(
            """
            SELECT id, username, created_at, last_login_at, status, must_change_password
            FROM users
            WHERE role = 'user'
            ORDER BY created_at DESC, id DESC
            """
        ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "username": str(row["username"]),
                "created_at": row["created_at"],
                "last_login_at": row["last_login_at"],
                "status": str(row["status"]),
                "must_change_password": bool(row["must_change_password"]),
            }
            for row in rows
        ]
    finally:
        connection.close()


def reset_user_password(
    db_path: Path | str,
    *,
    actor_user_id: int,
    target_user_id: int,
) -> dict[str, Any]:
    connection = connect_database(db_path)
    try:
        row = connection.execute(
            "SELECT id, username, role, status FROM users WHERE id = ?",
            (target_user_id,),
        ).fetchone()
        if row is None or row["role"] != "user":
            raise AdminError("只能重置普通用户密码。")
        if row["status"] != "active":
            raise AdminError("该账号不是可重置的活跃账号。")
        with connection:
            connection.execute(
                """
                UPDATE users
                SET password_hash = ?, must_change_password = 1
                WHERE id = ? AND role = 'user' AND status = 'active'
                """,
                (hash_password(DEFAULT_ADMIN_PASSWORD), target_user_id),
            )
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (target_user_id,))
            record_operation(
                connection,
                actor_user_id=actor_user_id,
                action="重置用户密码",
                target_type="user",
                target_id=str(target_user_id),
                detail={"username": str(row["username"]), "must_change_password": True},
            )
        return {
            "id": int(row["id"]),
            "username": str(row["username"]),
            "must_change_password": True,
        }
    finally:
        connection.close()


def ensure_public_document(db_path: Path | str, document_id: int) -> None:
    """确认目标是公共资料，避免管理员接口越过私人资料边界。"""

    connection = connect_database(db_path)
    try:
        row = connection.execute(
            "SELECT is_private FROM knowledge_documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if row is None or int(row["is_private"]) != 0:
            raise AdminError("只能管理公共资料。")
    finally:
        connection.close()


def list_public_documents(
    db_path: Path | str,
    *,
    query: str = "",
    module: str = "",
    content_status: str = "",
    validity_status: str = "",
    processing_status: str = "",
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    clauses = ["d.is_private = 0"]
    params: list[Any] = []
    if query.strip():
        clauses.append("(d.title LIKE ? OR d.source_label LIKE ? OR d.stable_key LIKE ?)")
        pattern = f"%{query.strip()}%"
        params.extend([pattern, pattern, pattern])
    for field, value in (
        ("d.module", module),
        ("d.content_status", content_status),
        ("d.validity_status", validity_status),
        ("d.processing_status", processing_status),
    ):
        if value.strip():
            clauses.append(f"{field} = ?")
            params.append(value.strip())
    where = " AND ".join(clauses)
    connection = connect_database(db_path)
    try:
        total = connection.execute(
            f"SELECT COUNT(*) FROM knowledge_documents d WHERE {where}", params
        ).fetchone()[0]
        rows = connection.execute(
            f"""
            SELECT d.id, d.stable_key, d.title, d.module, d.source_label, d.source_url,
                   d.published_date, d.publisher_department, d.content_status,
                   d.validity_status, d.processing_status, d.version, d.updated_at,
                   (SELECT COUNT(*) FROM knowledge_chunks c WHERE c.document_id = d.id) AS chunk_count,
                   (SELECT t.error_message FROM processing_tasks t
                    WHERE t.document_id = d.id ORDER BY t.id DESC LIMIT 1) AS error_message
            FROM knowledge_documents d
            WHERE {where}
            ORDER BY d.updated_at DESC, d.id DESC
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).fetchall()
        items = [
            {
                "id": int(row["id"]),
                "stable_key": str(row["stable_key"]),
                "title": str(row["title"]),
                "module": str(row["module"]),
                "source_label": str(row["source_label"] or ""),
                "source_url": row["source_url"],
                "published_date": row["published_date"],
                "publisher_department": row["publisher_department"],
                "content_status": str(row["content_status"]),
                "validity_status": str(row["validity_status"]),
                "processing_status": str(row["processing_status"]),
                "version": row["version"],
                "updated_at": row["updated_at"],
                "chunk_count": int(row["chunk_count"]),
                "error_message": row["error_message"],
            }
            for row in rows
        ]
        return {"items": items, "total": int(total), "limit": limit, "offset": offset}
    finally:
        connection.close()


def list_operation_logs(db_path: Path | str, *, limit: int = 100) -> list[dict[str, Any]]:
    connection = connect_database(db_path)
    try:
        with connection:
            _cleanup_old_logs(connection)
        rows = connection.execute(
            """
            SELECT l.id, l.action, l.target_type, l.target_id, l.detail_json,
                   l.created_at, COALESCE(u.username, '系统处理队列') AS actor_username
            FROM operation_logs l
            LEFT JOIN users u ON u.id = l.actor_user_id
            ORDER BY l.created_at DESC, l.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                detail = json.loads(row["detail_json"] or "{}")
            except json.JSONDecodeError:
                detail = {}
            result.append(
                {
                    "id": int(row["id"]),
                    "action": str(row["action"]),
                    "target_type": str(row["target_type"]),
                    "target_id": row["target_id"],
                    "detail": detail,
                    "created_at": row["created_at"],
                    "actor_username": row["actor_username"],
                }
            )
        return result
    finally:
        connection.close()


def list_processing_tasks(db_path: Path | str, *, limit: int = 100) -> list[dict[str, Any]]:
    connection = connect_database(db_path)
    try:
        rows = connection.execute(
            """
            SELECT t.id, t.document_id, d.title, d.stable_key, t.task_type,
                   t.status, t.error_message, t.created_at, t.finished_at
            FROM processing_tasks t
            JOIN knowledge_documents d ON d.id = t.document_id
            WHERE d.is_private = 0
            ORDER BY t.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "document_id": int(row["document_id"]),
                "title": str(row["title"]),
                "stable_key": str(row["stable_key"]),
                "task_type": str(row["task_type"]),
                "status": str(row["status"]),
                "error_message": row["error_message"],
                "created_at": row["created_at"],
                "finished_at": row["finished_at"],
            }
            for row in rows
        ]
    finally:
        connection.close()


def public_source_path(
    db_path: Path | str,
    source_root: Path,
    document_id: int,
) -> Path:
    connection = connect_database(db_path)
    try:
        row = connection.execute(
            "SELECT stable_key FROM knowledge_documents WHERE id = ? AND is_private = 0",
            (document_id,),
        ).fetchone()
        if row is None:
            raise AdminError("找不到公共资料。")
        stable_key = str(row["stable_key"])
    finally:
        connection.close()
    prefixes = ("public:markdown:", "public:file:")
    prefix = next((value for value in prefixes if stable_key.startswith(value)), None)
    if prefix is None:
        raise AdminError("该资料没有可重试的本地公共文件路径。")
    candidate = (source_root / Path(stable_key.removeprefix(prefix))).resolve()
    try:
        candidate.relative_to(source_root.resolve())
    except ValueError as exc:
        raise AdminError("资料路径不在公共资料根目录内。") from exc
    return candidate
