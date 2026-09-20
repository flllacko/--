from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.processing.metadata import MetadataError, update_document_metadata
from app.processing.processor import process_markdown_file


SOURCE = """# 元信息测试

## 资料信息

- 资料编号：TEST-META-001
- 所属模块：学生事务
- 内容状态：可直接回答
- 时效状态：有效
- 来源文字：原始来源
- 原文链接：

## 官方正文

这是一条用于元信息编辑测试的官方正文。
"""


class MetadataTest(unittest.TestCase):
    def test_admin_can_edit_metadata_without_reprocessing_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "资料正文"
            root.joinpath("学生事务").mkdir(parents=True)
            source = root / "学生事务" / "TEST-META-001 元信息测试.md"
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

            before = sqlite3.connect(database)
            try:
                before_count = before.execute(
                    "SELECT COUNT(*) FROM knowledge_chunks WHERE document_id = ?",
                    (result.document_id,),
                ).fetchone()[0]
            finally:
                before.close()

            updated = update_document_metadata(
                database,
                int(result.document_id),
                title="管理员修订后的标题",
                source_label="学生事务办公室",
                source_url="",
                published_date="2026-09-17",
                publisher_department="学生事务办公室",
            )

            self.assertEqual(updated.title, "管理员修订后的标题")
            self.assertIsNone(updated.source_url)
            after = sqlite3.connect(database)
            try:
                after_count = after.execute(
                    "SELECT COUNT(*) FROM knowledge_chunks WHERE document_id = ?",
                    (result.document_id,),
                ).fetchone()[0]
                source_record = after.execute(
                    "SELECT label, url FROM source_records WHERE document_id = ?",
                    (result.document_id,),
                ).fetchone()
            finally:
                after.close()

            self.assertEqual(after_count, before_count)
            self.assertEqual(source_record, ("学生事务办公室", None))

    def test_metadata_validation_rejects_invalid_status_and_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "app.db"
            root = Path(directory) / "资料正文" / "学生事务"
            root.mkdir(parents=True)
            source = root / "TEST-META-001 元信息测试.md"
            source.write_text(SOURCE, encoding="utf-8")
            with patch("app.processing.processor.embed_texts", return_value=[[0.1, 0.2, 0.3]]):
                result = process_markdown_file(
                    source,
                    source_root=root.parent,
                    db_path=database,
                    index_path=Path(directory) / "index",
                )

            with self.assertRaises(MetadataError):
                update_document_metadata(database, int(result.document_id), content_status="不确定")
            with self.assertRaises(MetadataError):
                update_document_metadata(database, int(result.document_id), source_url="ftp://example.com")


if __name__ == "__main__":
    unittest.main()
