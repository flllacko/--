from __future__ import annotations

import json
import unittest

from app.qa.agentic import keep_relevant_hits, merge_retrievals, review_evidence


def retrieval(question: str, hits: list[dict[str, object]]) -> dict[str, object]:
    return {
        "question": question,
        "retrieval": {"keyword_terms": ["休学"], "core_terms": ["休学"], "hit_count": len(hits)},
        "hits": hits,
    }


class AgenticDecisionTests(unittest.TestCase):
    def test_model_can_answer_or_clarify_without_topic_keywords(self) -> None:
        value = retrieval("学校办理时间是什么时候？", [])
        def fake_chat(messages, base_url, model, **kwargs):
            self.assertIn("问题表意清楚但资料不足不是歧义", messages[0]["content"])
            return '{"action":"clarify","query":"","clarification":"你想问休学还是复学的办理时间？"}'
        decision = review_evidence(
            value["question"], value, base_url="http://unused", model="test", chat_caller=fake_chat,
        )
        self.assertEqual(decision["action"], "clarify")
        self.assertIn("休学还是复学", decision["clarification"])

    def test_retry_preserves_year_and_has_one_round_limit(self) -> None:
        value = retrieval("2027届毕业生就业手续怎么办？", [])
        def fake_chat(messages, base_url, model, **kwargs):
            return '{"action":"retry","query":"2027届毕业生就业手续办理通知","clarification":""}'
        decision = review_evidence(
            value["question"], value, base_url="http://unused", model="test", chat_caller=fake_chat,
        )
        self.assertEqual(decision["action"], "retry")
        second = review_evidence(
            value["question"], value, base_url="http://unused", model="test", allow_retry=False,
            chat_caller=fake_chat,
        )
        self.assertEqual(second["action"], "answer")
        def changed_year(messages, base_url, model, **kwargs):
            return '{"action":"retry","query":"2026届毕业生就业手续办理通知","clarification":""}'
        unsafe = review_evidence(
            value["question"], value, base_url="http://unused", model="test", chat_caller=changed_year,
        )
        self.assertEqual(unsafe["action"], "answer")

    def test_invalid_review_falls_back_to_answer(self) -> None:
        decision = review_evidence(
            "休学需要材料吗？", retrieval("休学需要材料吗？", []),
            base_url="http://unused", model="test",
            chat_caller=lambda *args, **kwargs: "not json",
        )
        self.assertEqual(decision["action"], "answer")

    def test_retrieval_merge_renumbers_and_deduplicates_sources(self) -> None:
        first = retrieval("休学申请", [
            {"rank": 1, "document_id": 1, "chunk_index": 0, "text": "第一段"},
            {"rank": 2, "document_id": 2, "chunk_index": 0, "text": "第二段"},
        ])
        retry = retrieval("休学材料", [
            {"rank": 1, "document_id": 2, "chunk_index": 0, "text": "第二段"},
            {"rank": 2, "document_id": 3, "chunk_index": 1, "text": "第三段"},
        ])
        retry["retrieval"]["keyword_terms"] = ["材料"]
        merged = merge_retrievals(first, retry, limit=4)
        self.assertEqual([hit["rank"] for hit in merged["hits"]], [1, 2, 3])
        self.assertEqual([hit["document_id"] for hit in merged["hits"]], [2, 1, 3])
        self.assertEqual(merged["retrieval"]["keyword_terms"], ["休学", "材料"])
        self.assertEqual(merged["question"], "休学申请")

    def test_only_directly_relevant_hits_are_sent_to_generation(self) -> None:
        value = retrieval("休学需要哪些材料？", [
            {"rank": 1, "title": "休学审批表", "text": "填写休学审批表", "semantic_distance": 0.4},
            {"rank": 2, "title": "复学流程", "text": "复学时交医院证明", "semantic_distance": 0.5},
            {"rank": 3, "title": "军人资助", "text": "服兵役资助政策", "semantic_distance": 0.6},
        ])
        def fake_chat(messages, base_url, model, **kwargs):
            self.assertEqual(json.loads(messages[1]["content"])["asked_facets"], ["materials"])
            return '{"action":"answer","query":"","clarification":"","relevant_ranks":[1]}'
        decision = review_evidence(
            value["question"], value, base_url="http://unused", model="test",
            facets=["materials"], chat_caller=fake_chat,
        )
        narrowed = keep_relevant_hits(value, decision["relevant_ranks"])
        self.assertEqual([hit["title"] for hit in narrowed["hits"]], ["休学审批表"])
        self.assertEqual(narrowed["retrieval"]["hit_count"], 1)


if __name__ == "__main__":
    unittest.main()
