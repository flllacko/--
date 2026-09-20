from __future__ import annotations

"""P7 历史会话的持久化服务。

回答正文和来源片段跟随 assistant 消息保存为 JSON，SQLite 是唯一业务真源；
前端只通过这些接口读取自己的会话，不能读取其他账号的历史。
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.auth import connect_database


class ConversationError(ValueError):
    pass


class ConversationNotFoundError(ConversationError):
    pass


class ConversationPermissionError(ConversationError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _title_from_question(question: str) -> str:
    compact = " ".join(str(question).strip().split())
    if not compact:
        return "新会话"
    return compact[:36] + ("…" if len(compact) > 36 else "")


def _json_text(value: object, default: object) -> str:
    return json.dumps(value if value is not None else default, ensure_ascii=False)


def _owned_conversation(
    connection: sqlite3.Connection,
    user_id: int,
    conversation_id: int,
) -> sqlite3.Row:
    row = connection.execute(
        """
        SELECT * FROM conversations
        WHERE id = ? AND user_id = ? AND is_deleted = 0
        """,
        (conversation_id, user_id),
    ).fetchone()
    if row is None:
        raise ConversationPermissionError("找不到属于当前账号的会话。")
    return row


def ensure_conversation_owner(
    db_path: Path | str,
    *,
    user_id: int,
    conversation_id: int,
) -> None:
    connection = connect_database(db_path)
    try:
        _owned_conversation(connection, user_id, conversation_id)
    finally:
        connection.close()


def save_completed_answer(
    db_path: Path | str,
    *,
    user_id: int,
    conversation_id: int | None,
    question: str,
    answer: dict[str, Any],
    answer_status: str,
    sources: list[dict[str, Any]],
    identity: str,
) -> int:
    question_text = str(question).strip()
    if not question_text:
        raise ConversationError("不能保存空问题。")
    now = _now()
    answer_record = dict(answer)
    answer_record["answer_status"] = answer_status
    connection = connect_database(db_path)
    try:
        connection.execute("BEGIN")
        if conversation_id is None:
            cursor = connection.execute(
                """
                INSERT INTO conversations (user_id, title, identity, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, _title_from_question(question_text), identity, now, now),
            )
            target_id = int(cursor.lastrowid)
        else:
            current = _owned_conversation(connection, user_id, conversation_id)
            target_id = int(current["id"])
            connection.execute(
                "UPDATE conversations SET identity = ?, updated_at = ? WHERE id = ?",
                (identity, now, target_id),
            )
            if str(current["title"]) == "新会话":
                connection.execute(
                    "UPDATE conversations SET title = ? WHERE id = ?",
                    (_title_from_question(question_text), target_id),
                )

        connection.execute(
            """
            INSERT INTO messages
                (conversation_id, role, content, status, answer_json, sources_json, created_at)
            VALUES (?, 'user', ?, 'completed', NULL, '[]', ?)
            """,
            (target_id, question_text, now),
        )
        connection.execute(
            """
            INSERT INTO messages
                (conversation_id, role, content, status, answer_json, sources_json, created_at)
            VALUES (?, 'assistant', ?, 'completed', ?, ?, ?)
            """,
            (
                target_id,
                str(answer_record.get("conclusion") or ""),
                _json_text(answer_record, {}),
                _json_text(sources, []),
                now,
            ),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, target_id)
        )
        connection.commit()
        return target_id
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def import_guest_entries(
    db_path: Path | str,
    *,
    user_id: int,
    entries: Iterable[dict[str, Any]],
    identity: str,
) -> int:
    materialized = list(entries)
    if not materialized:
        raise ConversationError("没有可转入历史的临时会话。")
    first_question = str(materialized[0].get("question") or "").strip()
    if not first_question:
        raise ConversationError("临时会话缺少问题。")
    now = _now()
    connection = connect_database(db_path)
    try:
        connection.execute("BEGIN")
        cursor = connection.execute(
            """
            INSERT INTO conversations (user_id, title, identity, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, _title_from_question(first_question), identity, now, now),
        )
        conversation_id = int(cursor.lastrowid)
        inserted = 0
        for entry in materialized:
            question = str(entry.get("question") or "").strip()
            answer = entry.get("answer")
            if not question or not isinstance(answer, dict):
                continue
            sources = entry.get("sources")
            if not isinstance(sources, list):
                sources = []
            status = str(entry.get("answer_status") or "completed")
            stamp = _now()
            connection.execute(
                """
                INSERT INTO messages
                    (conversation_id, role, content, status, answer_json, sources_json, created_at)
                VALUES (?, 'user', ?, 'completed', NULL, '[]', ?)
                """,
                (conversation_id, question, stamp),
            )
            answer_record = dict(answer)
            answer_record["answer_status"] = status
            connection.execute(
                """
                INSERT INTO messages
                    (conversation_id, role, content, status, answer_json, sources_json, created_at)
                VALUES (?, 'assistant', ?, 'completed', ?, ?, ?)
                """,
                (
                    conversation_id,
                    str(answer_record.get("conclusion") or ""),
                    _json_text(answer_record, {}),
                    _json_text(sources, []),
                    stamp,
                ),
            )
            inserted += 1
        if not inserted:
            raise ConversationError("临时会话没有完整回答，未转入历史。")
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), conversation_id)
        )
        connection.commit()
        return conversation_id
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def list_conversations(
    db_path: Path | str,
    *,
    user_id: int,
    query: str = "",
) -> list[dict[str, Any]]:
    connection = connect_database(db_path)
    try:
        pattern = f"%{query.strip()}%"
        rows = connection.execute(
            """
            SELECT c.*, COUNT(m.id) AS message_count
            FROM conversations AS c
            LEFT JOIN messages AS m ON m.conversation_id = c.id
            WHERE c.user_id = ? AND c.is_deleted = 0 AND c.title LIKE ?
            GROUP BY c.id
            ORDER BY c.is_pinned DESC, c.updated_at DESC
            """,
            (user_id, pattern),
        ).fetchall()
        today = datetime.now(timezone.utc).date()
        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                date = datetime.fromisoformat(str(row["updated_at"])).date()
            except ValueError:
                date = today
            day_delta = (today - date).days
            if day_delta == 0:
                group = "今天"
            elif day_delta == 1:
                group = "昨天"
            elif day_delta <= 7:
                group = "7天内"
            elif day_delta <= 30:
                group = "30天内"
            else:
                group = "更早"
            result.append(
                {
                    "id": int(row["id"]),
                    "title": str(row["title"]),
                    "identity": str(row["identity"]),
                    "is_pinned": bool(row["is_pinned"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "message_count": int(row["message_count"]),
                    "time_group": group,
                }
            )
        return result
    finally:
        connection.close()


def get_conversation(
    db_path: Path | str,
    *,
    user_id: int,
    conversation_id: int,
) -> dict[str, Any]:
    connection = connect_database(db_path)
    try:
        conversation = _owned_conversation(connection, user_id, conversation_id)
        rows = connection.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
        messages: list[dict[str, Any]] = []
        for row in rows:
            item: dict[str, Any] = {
                "id": int(row["id"]),
                "role": str(row["role"]),
                "content": str(row["content"]),
                "status": str(row["status"]),
                "created_at": row["created_at"],
            }
            if row["role"] == "assistant":
                try:
                    item["answer"] = json.loads(row["answer_json"] or "{}")
                except json.JSONDecodeError:
                    item["answer"] = {"conclusion": str(row["content"])}
                try:
                    item["sources"] = json.loads(row["sources_json"] or "[]")
                except json.JSONDecodeError:
                    item["sources"] = []
            messages.append(item)
        return {
            "id": int(conversation["id"]),
            "title": str(conversation["title"]),
            "identity": str(conversation["identity"]),
            "is_pinned": bool(conversation["is_pinned"]),
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
            "messages": messages,
        }
    finally:
        connection.close()


def update_conversation(
    db_path: Path | str,
    *,
    user_id: int,
    conversation_id: int,
    title: str | None = None,
    is_pinned: bool | None = None,
) -> dict[str, Any]:
    if title is not None:
        title = " ".join(title.strip().split())[:80]
        if not title:
            raise ConversationError("会话标题不能为空。")
    if title is None and is_pinned is None:
        raise ConversationError("没有需要更新的会话字段。")
    connection = connect_database(db_path)
    try:
        _owned_conversation(connection, user_id, conversation_id)
        updates: list[str] = []
        values: list[Any] = []
        if title is not None:
            updates.append("title = ?")
            values.append(title)
        if is_pinned is not None:
            updates.append("is_pinned = ?")
            values.append(1 if is_pinned else 0)
        updates.append("updated_at = ?")
        values.append(_now())
        values.append(conversation_id)
        connection.execute(
            f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?", values
        )
        connection.commit()
        row = connection.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        return {
            "id": int(row["id"]),
            "title": str(row["title"]),
            "identity": str(row["identity"]),
            "is_pinned": bool(row["is_pinned"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
    finally:
        connection.close()


def delete_conversations(
    db_path: Path | str,
    *,
    user_id: int,
    conversation_ids: Iterable[int],
) -> int:
    ids = sorted({int(value) for value in conversation_ids})
    if not ids:
        return 0
    connection = connect_database(db_path)
    try:
        placeholders = ",".join("?" for _ in ids)
        params = [user_id, *ids]
        rows = connection.execute(
            f"SELECT id FROM conversations WHERE user_id = ? AND is_deleted = 0 AND id IN ({placeholders})",
            params,
        ).fetchall()
        owned_ids = [int(row["id"]) for row in rows]
        if not owned_ids:
            raise ConversationPermissionError("没有找到属于当前账号的会话。")
        placeholders = ",".join("?" for _ in owned_ids)
        connection.execute("BEGIN")
        connection.execute(
            f"DELETE FROM messages WHERE conversation_id IN ({placeholders})", owned_ids
        )
        connection.execute(
            f"DELETE FROM conversations WHERE id IN ({placeholders})", owned_ids
        )
        connection.commit()
        return len(owned_ids)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
