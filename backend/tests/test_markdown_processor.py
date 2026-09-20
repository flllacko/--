from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.processing.markdown import chunk_markdown, clean_markdown
from app.processing.processor import process_markdown_file


SAMPLE_MARKDOWN = """# 测试资助政策

## 资料信息

- 资料编号：TEST-XSZZ-001
- 所属模块：学生资助
- 内容状态：可直接回答
- 时效状态：有效
- 来源文字：测试官方来源
- 发布部门：学生资助办公室
- 原文链接：[官方页面](https://example.com/policy)
- 发布日期：2026-09-17

## 官方正文

国家助学金按照公开政策分为三档。学生应以学校正式通知为准办理相关事项。
"""


class MarkdownProcessorTest(unittest.TestCase):
    def test_clean_and_chunk_preserve_content(self) -> None:
        cleaned = clean_markdown("标题\r\n\r\n\r\n正文\x00  ")
        self.assertEqual(cleaned, "标题\n\n正文")
        chunks = chunk_markdown("第一段\n\n第二段", max_chars=20, overlap=4)
        self.assertEqual(len(chunks), 1)
        self.assertIn("第二段", chunks[0].content)

    def test_process_and_skip_exact_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "资料正文"
            root.joinpath("学生资助").mkdir(parents=True)
            source = root / "学生资助" / "TEST-XSZZ-001 测试政策.md"
            source.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
            database = Path(directory) / "app.db"
            index = Path(directory) / "index"

            with patch(
                "app.processing.processor.embed_texts",
                return_value=[[0.1, 0.2, 0.3]],
            ):
                first = process_markdown_file(
                    source,
                    source_root=root,
                    db_path=database,
                    index_path=index,
                )
                second = process_markdown_file(
                    source,
                    source_root=root,
                    db_path=database,
                    index_path=index,
                )

            self.assertEqual(first.status, "处理成功")
            self.assertGreaterEqual(first.chunk_count, 1)
            self.assertTrue(second.duplicate)
            self.assertEqual(first.document_id, second.document_id)

            connection = sqlite3.connect(database)
            try:
                task_count = connection.execute("SELECT COUNT(*) FROM processing_tasks").fetchone()[0]
                chunk_count = connection.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()[0]
                source_url = connection.execute("SELECT url FROM source_records").fetchone()[0]
                metadata = connection.execute(
                    """
                    SELECT source_label, published_date, publisher_department
                    FROM knowledge_documents
                    WHERE id = ?
                    """,
                    (first.document_id,),
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(task_count, 1)
            self.assertEqual(chunk_count, first.chunk_count)
            self.assertEqual(source_url, "https://example.com/policy")
            self.assertEqual(metadata, ("测试官方来源", "2026-09-17", "学生资助办公室"))

    def test_missing_metadata_is_recorded_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "资料正文"
            root.joinpath("教务").mkdir(parents=True)
            source = root / "教务" / "empty.md"
            source.write_text("# 只有标题，没有资料信息", encoding="utf-8")
            database = Path(directory) / "app.db"

            result = process_markdown_file(source, source_root=root, db_path=database)

            self.assertEqual(result.status, "处理失败")
            self.assertIn("资料编号", result.error_message or "")
            connection = sqlite3.connect(database)
            try:
                status = connection.execute(
                    "SELECT processing_status FROM knowledge_documents WHERE id = ?",
                    (result.document_id,),
                ).fetchone()[0]
                task_status = connection.execute(
                    "SELECT status FROM processing_tasks WHERE document_id = ?",
                    (result.document_id,),
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(status, "处理失败")
            self.assertEqual(task_status, "处理失败")


if __name__ == "__main__":
    unittest.main()
