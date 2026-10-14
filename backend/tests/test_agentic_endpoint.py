from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from starlette.requests import Request

from app import main as main_module


def guest_request() -> Request:
    return Request({"type": "http", "method": "POST", "path": "/api/answer/stream",
                    "headers": [], "query_string": b""})


async def collect(response) -> str:
    return "".join([chunk async for chunk in response.body_iterator])


def events(response_text: str) -> list[tuple[str, dict[str, object]]]:
    parsed = []
    for block in response_text.split("\n\n"):
        if not block.strip():
            continue
        lines = block.splitlines()
        name = next((line[7:] for line in lines if line.startswith("event: ")), "")
        data = next((line[6:] for line in lines if line.startswith("data: ")), "")
        if name and data:
            parsed.append((name, json.loads(data)))
    return parsed


class AgenticEndpointTests(unittest.TestCase):
    def test_clarification_stream_does_not_search_or_invent_sources(self) -> None:
        plan = {"mode": "school", "action": "clarify", "depth": "specific", "facets": [],
                "query": "学校办理时间", "clarification": "你想问哪项业务的办理时间？", "history": []}
        with patch.object(main_module, "_dialogue_plan", return_value=plan), patch.object(
            main_module, "search_knowledge", side_effect=AssertionError("不应检索")
        ):
            response = main_module.answer_stream(
                main_module.SearchRequest(question="学校办理时间是什么时候？"), guest_request(),
            )
            stream = events(asyncio.run(collect(response)))
        self.assertEqual(next(data["text"] for name, data in stream if name == "answer_delta"),
                         "你想问哪项业务的办理时间？")
        final = next(data for name, data in stream if name == "final")
        self.assertEqual(final["answer"]["conclusion"], "你想问哪项业务的办理时间？")
        self.assertEqual(final["sources"], [])

    def test_school_stream_conclusion_matches_final_and_sources(self) -> None:
        plan = {"mode": "school", "action": "answer", "depth": "overview", "facets": [],
                "query": "我想办休学", "clarification": "", "history": []}
        retrieval = {
            "question": "我想办休学", "original_question": "我想办休学",
            "answer_depth": "overview", "requested_facets": [],
            "retrieval": {"hit_count": 1, "keyword_terms": ["休学"], "core_terms": ["休学"]},
            "hits": [{"rank": 1, "document_id": 1, "chunk_index": 0,
                      "title": "休学审批表", "text": "申请休学填写休学审批表。",
                      "semantic_distance": 0.3, "match_types": ["semantic"]}],
        }
        raw = ('{"conclusion":"休学申请需要填写休学审批表。",'
               '"conditions":[],"materials":["休学审批表"],"steps":[],"time":"",'
               '"notes":[],"follow_up_questions":[],"source_ranks":[1]}')
        with patch.object(main_module, "_dialogue_plan", return_value=plan), patch.object(
            main_module, "_school_retrieval", return_value=(retrieval, "")
        ), patch.object(main_module, "stream_ollama_chat", return_value=iter([raw[:45], raw[45:]])):
            response = main_module.answer_stream(
                main_module.SearchRequest(question="我想办休学"), guest_request(),
            )
            stream = events(asyncio.run(collect(response)))
        raw_stream = "".join(data["text"] for name, data in stream if name == "delta")
        final = next(data for name, data in stream if name == "final")
        self.assertEqual(json.loads(raw_stream)["conclusion"], final["answer"]["conclusion"])
        self.assertEqual(final["answer"]["materials"], [])
        self.assertEqual([item["title"] for item in final["sources"]], ["休学审批表"])

    def test_retry_keeps_original_scope_and_private_owner(self) -> None:
        queries: list[tuple[str, str, int | None, bool]] = []
        def fake_search(query, **kwargs):
            queries.append((query, kwargs["scope"], kwargs["owner_user_id"], kwargs["include_private"]))
            return {"question": query, "hits": [], "retrieval": {"keyword_terms": [], "core_terms": []}}
        decisions = [
            {"action": "retry", "query": "2027届就业手续官方说明", "clarification": "", "relevant_ranks": []},
            {"action": "answer", "query": "", "clarification": "", "relevant_ranks": []},
        ]
        with patch.object(main_module, "search_knowledge", side_effect=fake_search), patch.object(
            main_module, "review_evidence", side_effect=decisions
        ):
            result, clarification = main_module._school_retrieval(
                main_module.SearchRequest(question="2027届就业手续怎么办？", use_private_knowledge=True),
                {"id": 7, "role": "user"}, "2027届就业手续怎么办？", "specific", ["steps"],
            )
        self.assertEqual(clarification, "")
        self.assertEqual(len(queries), 2)
        self.assertEqual([item[1:] for item in queries], [("current", 7, True), ("current", 7, True)])
        self.assertEqual(result["original_question"], "2027届就业手续怎么办？")

    def test_stream_repair_clears_provisional_answer_before_final(self) -> None:
        plan = {"mode": "school", "action": "answer", "depth": "specific", "facets": [],
                "query": "休学", "clarification": "", "history": []}
        retrieval = {"question": "休学", "hits": [
            {"rank": 1, "document_id": 1, "chunk_index": 0, "title": "休学规定",
             "text": "休学填写审批表。", "semantic_distance": 0.3, "match_types": ["semantic"]},
        ], "retrieval": {"hit_count": 1, "keyword_terms": ["休学"], "core_terms": ["休学"]}}
        raw = '{"conclusion":"初稿","conditions":[],"materials":[],"steps":[],"time":"","notes":[],"follow_up_questions":[],"source_ranks":[1]}'
        repaired = {"answer_status": "completed", "answer": {
            "conclusion": "校验后的回答", "conditions": [], "materials": [], "steps": [],
            "time": "", "notes": [], "follow_up_questions": [],
        }, "used_source_ranks": [1]}
        with patch.object(main_module, "_dialogue_plan", return_value=plan), patch.object(
            main_module, "_school_retrieval", return_value=(retrieval, "")
        ), patch.object(main_module, "stream_ollama_chat", return_value=iter([raw])), patch.object(
            main_module, "finalize_stream_answer", return_value=repaired
        ):
            response = main_module.answer_stream(
                main_module.SearchRequest(question="休学"), guest_request(),
            )
            names = [name for name, _ in events(asyncio.run(collect(response)))]
        self.assertLess(names.index("delta"), names.index("answer_reset"))
        self.assertLess(names.index("answer_reset"), names.index("final"))


if __name__ == "__main__":
    unittest.main()
