from __future__ import annotations

import unittest

from app.qa.generation import (
    _build_prompt,
    finalize_stream_answer,
    generate_answer,
    reliable_evidence_hits,
)


def _retrieval(distance: float | None, match_types: list[str], text: str = "国家助学金分为三档") -> dict[str, object]:
    return {
        "retrieval": {"keyword_terms": ["国家助学金", "助学", "三档"]},
        "hits": [
            {
                "rank": 1,
                "title": "国家助学金政策",
                "module": "学生资助",
                "text": text,
                "locator": "正文第1段",
                "source_label": "学校官方",
                "published_date": "2026-07-05",
                "semantic_distance": distance,
                "match_types": match_types,
            }
        ],
    }


class GenerationTest(unittest.TestCase):
    def test_prompt_prioritizes_direct_school_amount_over_general_range(self) -> None:
        question = "学校 2026 年公开的国家助学金分几档，每档多少钱？"
        hits = [
                {
                    "rank": 1,
                    "document_id": 1,
                    "chunk_index": 0,
                    "title": "国家统一标准",
                    "module": "学生资助",
                    "text": "高校可以在每年 2500—5000 元范围内自主确定。",
                    "locator": "正文",
                    "source_label": "教育部",
                    "source_url": "https://www.moe.gov.cn/policy.html",
                    "published_date": "2024-10-25",
                    "semantic_distance": 0.4,
                    "match_types": ["keyword", "semantic"],
                },
                {
                    "rank": 2,
                    "document_id": 2,
                    "chunk_index": 0,
                    "title": "学校 2026 版资助政策简介",
                    "module": "学生资助",
                    "text": "国家助学金 4900 元、3700 元、2500 元/人·年，共三档。",
                    "locator": "政策表格",
                    "source_label": "成都理工大学工程技术学院官方",
                    "source_url": "https://zdh.cdutetc.cn/policy.html",
                    "published_date": "2026-07-06",
                    "semantic_distance": 0.4,
                    "match_types": ["keyword", "semantic"],
                },
            ]
        prompt = _build_prompt(question, hits)
        system = prompt[0]["content"]
        user = prompt[1]["content"].replace(" ", "")
        self.assertIn("学校具体资料优先于只给全国一般范围的资料", system)
        self.assertIn("不能据此前者回答为‘未公开’", system)
        self.assertIn("4900元、3700元、2500元/人·年", user)
        self.assertNotIn("2500—5000元范围内自主确定", user)

        retrieval = {
            "question": question,
            "retrieval": {
                "keyword_terms": ["国家助学金", "每档多少钱", "2026"],
                "core_terms": ["国家助学金", "每档多少钱"],
            },
            "hits": hits,
        }
        preferred = reliable_evidence_hits(retrieval)
        self.assertEqual([hit["rank"] for hit in preferred], [2])

    def test_prompt_surfaces_multi_part_time_and_fee_evidence(self) -> None:
        prompt = _build_prompt(
            "2026 年四川专升本什么时候报名、缴费和打印准考证？",
            [
                {
                    "rank": 1,
                    "document_id": 10,
                    "chunk_index": 4,
                    "title": "专升本报名通知",
                    "module": "招生",
                    "text": "普通高校专升本全省统一网上报名及缴费时间安排在 2026 年 1 月 5 日至 11 日，每日 9:00—22:00。",
                    "locator": "正文",
                    "source_label": "学校官方",
                    "published_date": "2025-12-25",
                    "semantic_distance": 0.4,
                    "match_types": ["keyword", "semantic"],
                },
                {
                    "rank": 2,
                    "document_id": 10,
                    "chunk_index": 5,
                    "title": "专升本报名通知",
                    "module": "招生",
                    "text": "网上缴费。普通高校专升本报名考试费为 80 元/生。",
                    "locator": "正文",
                    "source_label": "学校官方",
                    "published_date": "2025-12-25",
                    "semantic_distance": 0.4,
                    "match_types": ["keyword", "semantic"],
                },
            ],
        )
        content = prompt[1]["content"].replace(" ", "")
        self.assertIn("2026年1月5日至11日", content)
        self.assertIn("报名考试费为80元/生", content)

    def test_structured_answer_is_parsed_and_sources_are_bound(self) -> None:
        calls: list[object] = []

        def fake_chat(messages, base_url, model, timeout):
            calls.append((messages, base_url, model, timeout))
            return '{"conclusion":"国家助学金分为三档。","conditions":[],"materials":[],"steps":[],"time":"","notes":[],"follow_up_questions":["问题1","问题2","问题3"],"source_ranks":[1]}'

        result = generate_answer(
            "国家助学金分几档？",
            _retrieval(0.5, ["semantic", "keyword"]),
            base_url="http://unused",
            model="qwen3.5:9b",
            chat_caller=fake_chat,
        )
        self.assertEqual(result["answer_status"], "completed")
        self.assertEqual(result["answer"]["conclusion"], "国家助学金分为三档。")
        self.assertEqual(result["answer"]["follow_up_questions"], ["问题1", "问题2"])
        self.assertEqual(result["used_source_ranks"], [1])
        self.assertEqual(len(calls), 1)

    def test_broad_service_intent_gets_natural_guidance(self) -> None:
        def fake_chat(messages, base_url, model, timeout):
            self.assertIn("不要只用‘资料未提供’结束回答", messages[0]["content"])
            return '{"conclusion":"官方资料未提供完整办理流程。","conditions":[],"materials":[],"steps":[],"time":"","notes":[],"follow_up_questions":[],"source_ranks":[1]}'

        retrieval = _retrieval(
            0.5,
            ["semantic", "keyword"],
            text="休学申请需要按学校学籍管理规定办理。",
        )
        retrieval["question"] = "我想办休学"
        retrieval["original_question"] = "我想办休学"
        result = generate_answer(
            "我想办休学",
            retrieval,
            base_url="http://unused",
            model="qwen3.5:9b",
            chat_caller=fake_chat,
        )
        conclusion = result["answer"]["conclusion"]
        self.assertTrue(conclusion.startswith("可以，我来帮你核对休学办理。"))
        self.assertIn("为了避免一次给你堆太多信息", conclusion)
        self.assertIn("你想先了解申请条件、所需材料、办理流程，还是时间节点？", conclusion)
        self.assertEqual(result["used_source_ranks"], [])
        self.assertEqual(
            result["answer"]["follow_up_questions"],
            [
                "休学需要满足哪些条件？",
                "休学需要准备哪些材料？",
                "休学的办理流程是什么？",
                "休学有哪些时间要求？",
            ],
        )

    def test_original_broad_intent_survives_retrieval_query_rewrite(self) -> None:
        def fake_chat(messages, base_url, model, timeout):
            self.assertIn("用户只表达了想办理休学", messages[0]["content"])
            return '{"conclusion":"现有资料未列出完整流程。","conditions":[],"materials":[],"steps":[],"time":"","notes":[],"follow_up_questions":[],"source_ranks":[1]}'

        retrieval = _retrieval(
            0.5,
            ["semantic", "keyword"],
            text="休学申请按照学籍管理规定办理。",
        )
        retrieval["question"] = "办理休学的具体流程、材料和时间节点是什么？"
        retrieval["original_question"] = "我想办休学"
        result = generate_answer(
            retrieval["question"],
            retrieval,
            base_url="http://unused",
            model="qwen3.5:9b",
            chat_caller=fake_chat,
        )
        self.assertIn("你想先了解申请条件、所需材料、办理流程，还是时间节点？", result["answer"]["conclusion"])
        self.assertEqual(result["used_source_ranks"], [])

    def test_low_confidence_evidence_refuses_without_calling_model(self) -> None:
        calls = 0

        def fake_chat(*args, **kwargs):
            nonlocal calls
            calls += 1
            return "{}"

        result = generate_answer(
            "宿舍晚归怎么处理？",
            _retrieval(0.94, ["semantic"], text="课程修读与考核管理办法"),
            base_url="http://unused",
            model="qwen3.5:9b",
            chat_caller=fake_chat,
        )
        self.assertEqual(result["answer_status"], "no_reliable_evidence")
        self.assertLessEqual(len(result["answer"]["follow_up_questions"]), 2)
        self.assertEqual(calls, 0)

    def test_invalid_json_is_retried_once(self) -> None:
        calls = 0

        def fake_chat(messages, base_url, model, timeout):
            nonlocal calls
            calls += 1
            if calls == 1:
                return "我先解释一下这个问题："
            return '{"conclusion":"国家助学金分为三档。","conditions":[],"materials":[],"steps":[],"time":"","notes":[],"follow_up_questions":[],"source_ranks":[1]}'

        result = generate_answer(
            "国家助学金分几档？",
            _retrieval(0.5, ["semantic", "keyword"]),
            base_url="http://unused",
            model="qwen3.5:9b",
            chat_caller=fake_chat,
        )
        self.assertEqual(result["answer_status"], "completed")
        self.assertEqual(calls, 2)

    def test_invalid_json_uses_compact_third_attempt(self) -> None:
        calls = 0

        def fake_chat(messages, base_url, model, timeout):
            nonlocal calls
            calls += 1
            if calls < 3:
                return "无法按结构返回"
            return '{"conclusion":"国家助学金分为三档。","conditions":[],"materials":[],"steps":[],"time":"","notes":[],"follow_up_questions":[],"source_ranks":[1]}'

        result = generate_answer(
            "国家助学金分几档？",
            _retrieval(0.5, ["semantic", "keyword"]),
            base_url="http://unused",
            model="qwen3.5:9b",
            chat_caller=fake_chat,
        )
        self.assertEqual(result["answer_status"], "completed")
        self.assertEqual(calls, 3)

    def test_truncated_stream_uses_non_stream_repair(self) -> None:
        calls = 0

        def fake_chat(messages, base_url, model, timeout):
            nonlocal calls
            calls += 1
            return '{"conclusion":"复学需要提交复学申请表。","conditions":[],"materials":["复学申请表"],"steps":[],"time":"","notes":[],"follow_up_questions":[],"source_ranks":[1]}'

        result = finalize_stream_answer(
            "休学、复学需要准备什么材料？",
            '{"conclusion":"复学需要提交',
            _retrieval(0.5, ["semantic", "keyword"], text="复学申请表"),
            base_url="http://unused",
            model="qwen3.5:9b",
            chat_caller=fake_chat,
        )
        self.assertEqual(result["answer_status"], "completed")
        self.assertEqual(result["answer"]["materials"], ["复学申请表"])
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
