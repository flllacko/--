from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.auth import DEFAULT_ADMIN_PASSWORD, DEFAULT_ADMIN_USERNAME, hash_password


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _migrate_processing_task_cancelled_status(connection: sqlite3.Connection) -> None:
    """把 P4 已存在的任务表升级为支持排队取消。"""

    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'processing_tasks'"
    ).fetchone()
    definition = row[0] if row else ""
    if "已取消" in definition:
        return

    connection.execute("PRAGMA foreign_keys = OFF")
    try:
        connection.execute("BEGIN")
        connection.execute(
            """
            CREATE TABLE processing_tasks_migration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('排队中', '处理中', '处理成功', '处理失败', '已取消')),
                error_message TEXT,
                created_at TEXT NOT NULL,
                finished_at TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO processing_tasks_migration
                (id, document_id, task_type, status, error_message, created_at, finished_at)
            SELECT id, document_id, task_type, status, error_message, created_at, finished_at
            FROM processing_tasks
            """
        )
        connection.execute("DROP TABLE processing_tasks")
        connection.execute("ALTER TABLE processing_tasks_migration RENAME TO processing_tasks")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_status_created ON processing_tasks(status, created_at)"
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.execute("PRAGMA foreign_keys = ON")


def _ensure_document_metadata_columns(connection: sqlite3.Connection) -> None:
    existing = {
        row[1]
        for row in connection.execute("PRAGMA table_info(knowledge_documents)").fetchall()
    }
    columns = {
        "source_label": "TEXT NOT NULL DEFAULT ''",
        "published_date": "TEXT",
        "publisher_department": "TEXT",
        "version_group_key": "TEXT",
        "supersedes_document_id": "INTEGER REFERENCES knowledge_documents(id)",
    }
    for name, definition in columns.items():
        if name not in existing:
            connection.execute(
                f"ALTER TABLE knowledge_documents ADD COLUMN {name} {definition}"
            )


def _ensure_conversation_message_columns(connection: sqlite3.Connection) -> None:
    """为 P7 历史会话补充结构化回答和来源字段，不重建已有消息表。"""

    existing = {
        row[1]
        for row in connection.execute("PRAGMA table_info(messages)").fetchall()
    }
    columns = {
        "answer_json": "TEXT",
        "sources_json": "TEXT NOT NULL DEFAULT '[]'",
    }
    for name, definition in columns.items():
        if name not in existing:
            connection.execute(
                f"ALTER TABLE messages ADD COLUMN {name} {definition}"
            )


def _ensure_default_admin(connection: sqlite3.Connection) -> None:
    """首次初始化时建立唯一管理员；数据库中只保存哈希，不保存默认密码。"""

    row = connection.execute(
        "SELECT id FROM users WHERE username = ?",
        (DEFAULT_ADMIN_USERNAME,),
    ).fetchone()
    if row:
        return
    now = utc_now()
    connection.execute(
        """
        INSERT INTO users
            (username, password_hash, role, must_change_password, status, created_at)
        VALUES (?, ?, 'admin', 0, 'active', ?)
        """,
        (DEFAULT_ADMIN_USERNAME, hash_password(DEFAULT_ADMIN_PASSWORD), now),
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize_database(db_path: Path | None = None) -> Path:
    target = Path(db_path) if db_path else settings.sqlite_path
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        _migrate_processing_task_cancelled_status(connection)
        _ensure_document_metadata_columns(connection)
        _ensure_conversation_message_columns(connection)
        _ensure_default_admin(connection)
        connection.commit()
    finally:
        connection.close()
    return target
