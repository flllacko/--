from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.db.init_db import initialize_database


class SchemaTest(unittest.TestCase):
    def test_minimum_business_tables_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "test.db"
            initialize_database(database)
            connection = sqlite3.connect(database)
            try:
                names = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
            finally:
                connection.close()
            self.assertTrue(
                {
                    "users",
                    "conversations",
                    "messages",
                    "knowledge_documents",
                    "knowledge_chunks",
                    "source_records",
                    "processing_tasks",
                    "operation_logs",
                }.issubset(names)
            )


if __name__ == "__main__":
    unittest.main()
