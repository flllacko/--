from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal, Sequence

from app.db.init_db import initialize_database


KnowledgeScope = Literal["current", "historical", "navigation"]
VALID_CONTENT_STATUS = {"可直接回答", "历史可回答", "只能提供入口", "禁止作为依据"}
VALIDITY_STATUS = {"有效", "无效"}


class KnowledgeScopeError(ValueError):
    """知识库状态或检索范围不符合项目规则。"""


def validate_status_values(db_path: Path) -> list[int]:
    """返回状态不符合规则的资料 ID；不替管理员猜测或自动修复。"""

    initialize_database(db_path)
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT id
            FROM knowledge_documents
            WHERE is_private = 0
              AND content_status NOT IN (?, ?, ?, ?)
               OR validity_status NOT IN (?, ?)
            ORDER BY id
            """,
            (
                "可直接回答",
                "历史可回答",
                "只能提供入口",
                "禁止作为依据",
                "有效",
                "无效",
            ),
        ).fetchall()
        return [int(row[0]) for row in rows]
    finally:
        connection.close()


def document_ids_for_scope(
    db_path: Path,
    scope: KnowledgeScope,
    *,
    allow_invalid_history: bool = False,
) -> set[int]:
    """取得当前检索范围允许使用的资料 ID。

    `allow_invalid_history` 只供明确的历史问题使用，默认关闭，避免旧资料混入当前政策回答。
    """

    invalid_ids = validate_status_values(db_path)
    if invalid_ids:
        raise KnowledgeScopeError(f"存在未规范化的资料状态：{invalid_ids}")

    if scope == "current":
        content_statuses = ("可直接回答",)
        validity_statuses = ("有效",)
    elif scope == "historical":
        content_statuses = ("历史可回答",)
        validity_statuses = ("有效", "无效") if allow_invalid_history else ("有效",)
    elif scope == "navigation":
        content_statuses = ("只能提供入口",)
        validity_statuses = ("有效",)
    else:
        raise KnowledgeScopeError(f"未知知识库范围：{scope}")

    content_placeholders = ",".join("?" for _ in content_statuses)
    validity_placeholders = ",".join("?" for _ in validity_statuses)
    initialize_database(db_path)
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            f"""
            SELECT id
            FROM knowledge_documents
            WHERE processing_status = '处理成功'
              AND is_private = 0
              AND content_status IN ({content_placeholders})
              AND validity_status IN ({validity_placeholders})
            ORDER BY id
            """,
            (*content_statuses, *validity_statuses),
        ).fetchall()
        return {int(row[0]) for row in rows}
    finally:
        connection.close()


def private_document_ids_for_user(db_path: Path, user_id: int) -> set[int]:
    """只返回指定账号处理成功的私人资料 ID。"""

    initialize_database(db_path)
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT id
            FROM knowledge_documents
            WHERE owner_user_id = ?
              AND is_private = 1
              AND processing_status = '处理成功'
            ORDER BY id
            """,
            (user_id,),
        ).fetchall()
        return {int(row[0]) for row in rows}
    finally:
        connection.close()


def filter_vector_rows(rows: Sequence[dict], allowed_document_ids: set[int]) -> list[dict]:
    """按 SQLite 真源的允许 ID 过滤 LanceDB 结果。"""

    return [row for row in rows if int(row.get("document_id", -1)) in allowed_document_ids]
