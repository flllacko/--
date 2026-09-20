from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.processing.deletion import DeletionError, DeletionService
from app.processing.lance_store import has_document
from app.processing.processor import process_markdown_file


SOURCE = """# 删除测试资料

## 资料信息

- 资料编号：TEST-DELETE-001
- 所属模块：学生事务
- 内容状态：可直接回答
- 时效状态：有效
- 来源文字：测试官方来源

## 官方正文

这是一条用于验证删除边界的正文。
"""


class DeletionTest(unittest.TestCase):
    def test_requires_second_confirmation_and_deletes_all_materialized_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "资料正文"
            root.joinpath("学生事务").mkdir(parents=True)
            source = root / "学生事务" / "TEST-DELETE-001 删除测试资料.md"
            source.write_text(SOURCE, encoding="utf-8")
            database = base / "app.db"
            index = base / "index"

            with patch("app.processing.processor.embed_texts", return_value=[[0.1, 0.2, 0.3]]):
                result = process_markdown_file(
                    source,
                    source_root=root,
                    db_path=database,
                    index_path=index,
                )

            service = DeletionService(database, index, root)
            plan = service.create_plan(int(result.document_id))
            self.assertEqual(plan.source_path, source.resolve())
            with self.assertRaises(DeletionError):
                service.confirm_and_delete("wrong-token")

            deleted = service.confirm_and_delete(plan.confirmation_token)
            self.assertTrue(deleted.deleted_source_file)
            self.assertTrue(deleted.deleted_vector_rows)
            self.assertFalse(source.exists())
            self.assertFalse(has_document(index, int(result.document_id)))

            connection = sqlite3.connect(database)
            try:
                document_count = connection.execute(
                    "SELECT COUNT(*) FROM knowledge_documents WHERE id = ?",
                    (result.document_id,),
                ).fetchone()[0]
                chunk_count = connection.execute(
                    "SELECT COUNT(*) FROM knowledge_chunks WHERE document_id = ?",
                    (result.document_id,),
                ).fetchone()[0]
                task_count = connection.execute(
                    "SELECT COUNT(*) FROM processing_tasks WHERE document_id = ?",
                    (result.document_id,),
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual((document_count, chunk_count, task_count), (0, 0, 0))

    def test_path_outside_source_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "资料正文"
            root.mkdir()
            database = base / "app.db"
            service = DeletionService(database, base / "index", root)
            with self.assertRaises(DeletionError):
                service._resolve_source_path("public:markdown:学生事务/a.md", base / "outside.md")


if __name__ == "__main__":
    unittest.main()
