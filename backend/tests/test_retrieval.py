from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.db.init_db import initialize_database, utc_now
from app.processing.lance_store import sync_chunks
from app.processing.markdown import ParsedMarkdown, TextChunk
from app.qa.retrieval import _anchor_query_terms, search_knowledge


class RetrievalTest(unittest.TestCase):
    def test_business_anchors_keep_multi_part_admission_questions_on_topic(self) -> None:
        admission_anchors = _anchor_query_terms(
            "2026 年专升本学制多久，毕业拿什么证书，能不能转专业？"
        )
        cost_anchors = _anchor_query_terms("工程造价主要学哪些课程，毕业可以去哪些单位？")
        retired_soldier_anchors = _anchor_query_terms(
            "2026 年退役士兵批专升本有哪些专业、计划多少，2025 年各专业最高分和最低分是多少？"
        )
        major_mapping_anchors = _anchor_query_terms(
            "专科专业能对应哪个专升本专业，要考哪些科目？"
        )
        registration_anchors = _anchor_query_terms(
            "2026 年四川专升本什么时候报名、缴费和打印准考证？"
        )
        scholarship_combination_anchors = _anchor_query_terms(
            "国家奖学金和国家励志奖学金能不能同时获得，能不能同时申请国家助学金？"
        )

        self.assertIn("学士学位证书", admission_anchors)
        self.assertIn("转专业", admission_anchors)
        self.assertIn("就业单位", cost_anchors)
        self.assertIn("中核二二", cost_anchors)
        self.assertIn("2025年录取最低分", retired_soldier_anchors)
        self.assertIn("本专科专业对应指导目录", major_mapping_anchors)
        self.assertIn("考试科目", major_mapping_anchors)
        self.assertIn("报名考试费", registration_anchors)
        self.assertIn("网上报名", registration_anchors)
        self.assertIn("日程安排表", registration_anchors)
        self.assertIn("80 元/生", registration_anchors)
        self.assertIn("2026 年 1 月 5 日至 11 日", registration_anchors)
        self.assertIn("准考证", registration_anchors)
        self.assertIn("不能同时获得", scholarship_combination_anchors)

    def _insert_document(
        self,
        connection: sqlite3.Connection,
        *,
        stable_key: str,
        title: str,
        content: str,
        content_status: str = "可直接回答",
        validity_status: str = "有效",
    ) -> int:
        now = utc_now()
        cursor = connection.execute(
            """
            INSERT INTO knowledge_documents
                (stable_key, title, module, source_text, source_label, source_url,
                 content_status, validity_status, processing_status, created_at, updated_at)
            VALUES (?, ?, '学生资助', ?, '学校官方', 'https://example.com/source', ?, ?, '处理成功', ?, ?)
            """,
            (stable_key, title, content, content_status, validity_status, now, now),
        )
        document_id = int(cursor.lastrowid)
        connection.execute(
            """
            INSERT INTO knowledge_chunks
                (document_id, chunk_index, content, source_locator, created_at)
            VALUES (?, 0, ?, '正文第1段', ?)
            """,
            (document_id, content, now),
        )
        connection.commit()
        return document_id

    def test_search_merges_and_filters_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "app.db"
            index = root / "index"
            initialize_database(database)
            connection = sqlite3.connect(database)
            try:
                current_id = self._insert_document(
                    connection,
                    stable_key="public:current.md",
                    title="国家助学金政策",
                    content="国家助学金分为三档，按学校官方通知评审。",
                )
                blocked_id = self._insert_document(
                    connection,
                    stable_key="public:blocked.md",
                    title="内部草稿",
                    content="国家助学金内部草稿，不得作为依据。",
                    content_status="禁止作为依据",
                )
            finally:
                connection.close()

            for document_id, stable_key, title, content, status in [
                (current_id, "public:current.md", "国家助学金政策", "国家助学金分为三档", "可直接回答"),
                (blocked_id, "public:blocked.md", "内部草稿", "国家助学金内部草稿", "禁止作为依据"),
            ]:
                parsed = ParsedMarkdown(
                    title=title,
                    module="学生资助",
                    source_text=content,
                    source_url="https://example.com/source",
                    source_label="学校官方",
                    content_status=status,
                    validity_status="有效",
                    metadata={},
                )
                sync_chunks(
                    index,
                    document_id,
                    stable_key,
                    parsed,
                    [TextChunk(content=content, locator="正文第1段")],
                    [[1.0, 0.0]],
                    "test-fixture",
                )

            result = search_knowledge(
                "国家助学金分几档？",
                db_path=database,
                index_path=index,
                ollama_base_url="http://unused",
                embedding_model="test-fixture",
                embedder=lambda *args, **kwargs: [[1.0, 0.0]],
            )
            self.assertEqual(result["answer_status"], "retrieved_only")
            hits = result["hits"]
            self.assertTrue(hits)
            self.assertEqual(hits[0]["document_id"], current_id)
            self.assertNotIn(blocked_id, [hit["document_id"] for hit in hits])
            self.assertIn("keyword", hits[0]["match_types"])
            self.assertIn("semantic", hits[0]["match_types"])

    def test_search_can_use_keyword_hit_when_vector_candidates_are_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "app.db"
            initialize_database(database)
            connection = sqlite3.connect(database)
            try:
                document_id = self._insert_document(
                    connection,
                    stable_key="public:keyword.md",
                    title="补考报名通知",
                    content="补考报名需要在教务系统提交申请。",
                )
            finally:
                connection.close()

            result = search_knowledge(
                "补考报名",
                db_path=database,
                index_path=root / "missing-index",
                ollama_base_url="http://unused",
                embedding_model="test-fixture",
                embedder=lambda *args, **kwargs: [[1.0, 0.0]],
                vector_searcher=lambda *args, **kwargs: [],
            )
            self.assertEqual(result["hits"][0]["document_id"], document_id)
            self.assertEqual(result["hits"][0]["match_types"], ["keyword"])


if __name__ == "__main__":
    unittest.main()
