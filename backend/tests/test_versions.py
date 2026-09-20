from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.processing.versions import process_with_version_choice


OLD = """# 旧版资助通知

## 资料信息

- 资料编号：TEST-VERSION-OLD
- 所属模块：学生资助
- 内容状态：可直接回答
- 时效状态：有效
- 来源文字：测试官方来源
- 原文链接：[旧版](https://example.com/old)
- 发布日期：2026-01-01

## 官方正文

旧版政策正文，国家助学金为三档。
"""

NEW = """# 新版资助通知

## 资料信息

- 资料编号：TEST-VERSION-NEW
- 所属模块：学生资助
- 内容状态：可直接回答
- 时效状态：有效
- 来源文字：测试官方来源
- 原文链接：[新版](https://example.com/new)
- 发布日期：2026-09-17

## 官方正文

新版政策正文，国家助学金仍为三档但金额已更新。
"""


class VersionTest(unittest.TestCase):
    def test_exact_same_content_across_paths_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "资料正文"
            root.joinpath("学生资助").mkdir(parents=True)
            first_path = root / "学生资助" / "first.md"
            duplicate_path = root / "学生资助" / "duplicate.md"
            first_path.write_text(OLD, encoding="utf-8")
            duplicate_path.write_text(OLD, encoding="utf-8")
            database = base / "app.db"
            index = base / "index"

            with patch("app.processing.processor.embed_texts", return_value=[[0.1, 0.2, 0.3]]):
                from app.processing.processor import process_markdown_file

                first = process_markdown_file(
                    first_path,
                    source_root=root,
                    db_path=database,
                    index_path=index,
                )
                duplicate = process_with_version_choice(
                    duplicate_path,
                    relation="独立资料",
                    source_root=root,
                    db_path=database,
                    index_path=index,
                )

            self.assertTrue(duplicate.duplicate)
            self.assertEqual(duplicate.new_document_id, first.document_id)
            connection = sqlite3.connect(database)
            try:
                count = connection.execute("SELECT COUNT(*) FROM knowledge_documents").fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(count, 1)

    def test_old_version_stays_valid_until_new_version_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "资料正文"
            root.joinpath("学生资助").mkdir(parents=True)
            old_path = root / "学生资助" / "old.md"
            new_path = root / "学生资助" / "new.md"
            old_path.write_text(OLD, encoding="utf-8")
            new_path.write_text(NEW, encoding="utf-8")
            database = base / "app.db"
            index = base / "index"

            with patch("app.processing.processor.embed_texts", return_value=[[0.1, 0.2, 0.3]]):
                from app.processing.processor import process_markdown_file

                old_result = process_markdown_file(
                    old_path,
                    source_root=root,
                    db_path=database,
                    index_path=index,
                )
                version_result = process_with_version_choice(
                    new_path,
                    relation="旧资料新版本",
                    old_document_id=int(old_result.document_id),
                    source_root=root,
                    db_path=database,
                    index_path=index,
                )

            self.assertEqual(version_result.status, "处理成功")
            self.assertNotEqual(version_result.old_document_id, version_result.new_document_id)
            connection = sqlite3.connect(database)
            try:
                old_status = connection.execute(
                    "SELECT validity_status FROM knowledge_documents WHERE id = ?",
                    (old_result.document_id,),
                ).fetchone()[0]
                new_row = connection.execute(
                    """
                    SELECT validity_status, supersedes_document_id
                    FROM knowledge_documents WHERE id = ?
                    """,
                    (version_result.new_document_id,),
                ).fetchone()
            finally:
                connection.close()

            self.assertEqual(old_status, "无效")
            self.assertEqual(new_row, ("有效", old_result.document_id))

    def test_failed_new_version_does_not_disable_old_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "资料正文"
            root.joinpath("学生资助").mkdir(parents=True)
            old_path = root / "学生资助" / "old.md"
            bad_path = root / "学生资助" / "bad.md"
            old_path.write_text(OLD, encoding="utf-8")
            bad_path.write_text("# 新版但缺少资料编号", encoding="utf-8")
            database = base / "app.db"
            index = base / "index"

            with patch("app.processing.processor.embed_texts", return_value=[[0.1, 0.2, 0.3]]):
                from app.processing.processor import process_markdown_file

                old_result = process_markdown_file(
                    old_path,
                    source_root=root,
                    db_path=database,
                    index_path=index,
                )
                version_result = process_with_version_choice(
                    bad_path,
                    relation="旧资料新版本",
                    old_document_id=int(old_result.document_id),
                    source_root=root,
                    db_path=database,
                    index_path=index,
                )

            self.assertEqual(version_result.status, "处理失败")
            connection = sqlite3.connect(database)
            try:
                old_row = connection.execute(
                    "SELECT processing_status, validity_status FROM knowledge_documents WHERE id = ?",
                    (old_result.document_id,),
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(old_row, ("处理成功", "有效"))


if __name__ == "__main__":
    unittest.main()
