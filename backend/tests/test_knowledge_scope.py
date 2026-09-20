from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.db.init_db import initialize_database, utc_now
from app.processing.knowledge_scope import (
    KnowledgeScopeError,
    document_ids_for_scope,
    filter_vector_rows,
    validate_status_values,
)


class KnowledgeScopeTest(unittest.TestCase):
    def test_scope_filters_processing_content_and_validity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "app.db"
            initialize_database(database)
            connection = sqlite3.connect(database)
            try:
                now = utc_now()
                rows = [
                    ("current", "可直接回答", "有效"),
                    ("history", "历史可回答", "无效"),
                    ("nav", "只能提供入口", "有效"),
                    ("blocked", "禁止作为依据", "有效"),
                    ("failed", "可直接回答", "有效"),
                ]
                for title, content_status, validity_status in rows:
                    cursor = connection.execute(
                        """
                        INSERT INTO knowledge_documents
                            (stable_key, title, module, source_text, content_status,
                             validity_status, processing_status, created_at, updated_at)
                        VALUES (?, ?, '学生资助', '正文', ?, ?, ?, ?, ?)
                        """,
                        (
                            f"test:{title}",
                            title,
                            content_status,
                            validity_status,
                            "处理失败" if title == "failed" else "处理成功",
                            now,
                            now,
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO processing_tasks
                            (document_id, task_type, status, created_at, finished_at)
                        VALUES (?, 'test', ?, ?, ?)
                        """,
                        (
                            cursor.lastrowid,
                            "处理失败" if title == "failed" else "处理成功",
                            now,
                            now,
                        ),
                    )
                connection.commit()
            finally:
                connection.close()

            self.assertEqual(document_ids_for_scope(database, "current"), {1})
            self.assertEqual(document_ids_for_scope(database, "historical"), set())
            self.assertEqual(
                document_ids_for_scope(database, "historical", allow_invalid_history=True),
                {2},
            )
            self.assertEqual(document_ids_for_scope(database, "navigation"), {3})
            self.assertEqual(validate_status_values(database), [])

    def test_invalid_status_blocks_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "app.db"
            initialize_database(database)
            connection = sqlite3.connect(database)
            try:
                now = utc_now()
                connection.execute(
                    """
                    INSERT INTO knowledge_documents
                        (stable_key, title, module, source_text, content_status,
                         validity_status, processing_status, created_at, updated_at)
                    VALUES ('bad', 'bad', '学生资助', '正文', '待判断', '有效', '处理成功', ?, ?)
                    """,
                    (now, now),
                )
                connection.commit()
            finally:
                connection.close()

            with self.assertRaises(KnowledgeScopeError):
                document_ids_for_scope(database, "current")

    def test_vector_rows_are_filtered_by_sqlite_ids(self) -> None:
        rows = [
            {"document_id": 1, "text": "当前资料"},
            {"document_id": 2, "text": "旧资料"},
        ]
        self.assertEqual(filter_vector_rows(rows, {1}), [rows[0]])


if __name__ == "__main__":
    unittest.main()
