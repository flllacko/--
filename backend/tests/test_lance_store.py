from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.processing.lance_store import has_document, sync_chunks, vector_search
from app.processing.markdown import ParsedMarkdown, TextChunk


class LanceStoreTest(unittest.TestCase):
    def test_store_and_search_vector_fixture(self) -> None:
        parsed = ParsedMarkdown(
            title="测试资料",
            module="学生资助",
            source_text="测试正文",
            source_url="https://example.com",
            source_label="测试来源",
            content_status="可直接回答",
            validity_status="有效",
            metadata={},
        )
        chunks = [TextChunk(content="国家助学金三档", locator="正文")]
        fixture_vector = [[0.1, 0.2, 0.3]]

        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "index"
            count = sync_chunks(
                index_path,
                document_id=1,
                stable_key="public:markdown:test.md",
                parsed=parsed,
                chunks=chunks,
                embeddings=fixture_vector,
                embedding_model="test-fixture",
            )
            self.assertEqual(count, 1)
            self.assertTrue(has_document(index_path, 1))
            results = vector_search(index_path, fixture_vector[0], limit=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["document_id"], 1)


if __name__ == "__main__":
    unittest.main()
