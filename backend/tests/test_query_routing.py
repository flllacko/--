from __future__ import annotations

import unittest

from app.qa.query_routing import is_known_current_gap, resolve_scope


class QueryRoutingTest(unittest.TestCase):
    def test_employment_site_sections_use_navigation_scope(self) -> None:
        self.assertEqual(resolve_scope("学校就业网能看到哪些学生服务？", "current"), "navigation")
        self.assertEqual(resolve_scope("就业信息网有哪些栏目？", "current"), "navigation")
        self.assertEqual(resolve_scope("2027届毕业生就业手续怎么办？", "current"), "current")

    def test_explicit_historical_query_uses_historical_scope(self) -> None:
        self.assertEqual(
            resolve_scope("按 2023 年公开附件，申请在校外住宿需要哪些材料？", "current"),
            "historical",
        )

    def test_current_cohort_batch_query_uses_the_available_batch_record(self) -> None:
        self.assertEqual(
            resolve_scope("2026 级新生网上选宿舍怎么办？", "current"),
            "historical",
        )

    def test_program_faq_uses_historical_reference_scope(self) -> None:
        self.assertEqual(
            resolve_scope("智能建造专业主要学什么、学习难度怎么样，毕业能做什么？", "current"),
            "historical",
        )

    def test_admission_lookup_entry_uses_navigation_scope(self) -> None:
        self.assertEqual(
            resolve_scope("2026 年哪些批次可以在学校录取查询系统查，入口在哪里？", "current"),
            "navigation",
        )

    def test_current_grant_schedule_without_year_is_a_known_gap(self) -> None:
        self.assertTrue(is_known_current_gap("国家助学金什么时候发？"))
        self.assertFalse(is_known_current_gap("2026 年 6—7 月国家助学金没到账，应该找谁？"))

    def test_current_2026_admission_rules_stay_in_current_scope(self) -> None:
        self.assertEqual(
            resolve_scope("2026 年在四川的招生计划和专业学费在哪里看？", "current"),
            "current",
        )
        self.assertEqual(
            resolve_scope("2026 年专升本学制多久，毕业拿什么证书，能不能转专业？", "current"),
            "current",
        )
        self.assertEqual(
            resolve_scope(
                "2024年9月1日以后入伍，本专科生学费补偿或贷款代偿最高按什么标准？",
                "current",
            ),
            "current",
        )

    def test_2026_admission_guide_stays_historical(self) -> None:
        self.assertEqual(
            resolve_scope("2026 报考指南里学校有哪些院系和专业方向？", "current"),
            "historical",
        )

    def test_valid_2026_thesis_plan_stays_in_current_scope(self) -> None:
        self.assertEqual(
            resolve_scope(
                "按电子信息与计算机工程系 2026 届论文计划，开题、答辩和论文归档大致在什么时候？",
                "current",
            ),
            "current",
        )

    def test_specific_2026_defense_batch_uses_historical_notice(self) -> None:
        self.assertEqual(
            resolve_scope(
                "电计系 2026 届毕业论文什么时候一辩、二辩，答辩前要准备什么，迟到会怎样？",
                "current",
            ),
            "historical",
        )

    def test_2026_retired_soldier_rules_use_historical_provincial_notice(self) -> None:
        self.assertEqual(
            resolve_scope(
                "2026 年四川专升本有哪些考生类别，退役大学生士兵什么条件可以免试？",
                "current",
            ),
            "historical",
        )

    def test_2026_modern_secretary_directory_uses_historical_scope(self) -> None:
        self.assertEqual(
            resolve_scope(
                "我读的是现代文秘，2026 年专升本能对应哪些本科专业，考试科目是什么？",
                "current",
            ),
            "historical",
        )

    def test_associated_major_directory_uses_historical_scope(self) -> None:
        self.assertEqual(
            resolve_scope("2025 专升本专业对应目录能不能直接回答 2026 或 2027？", "current"),
            "historical",
        )

    def test_specific_current_rules_are_not_captured_by_broad_history_rules(self) -> None:
        self.assertEqual(
            resolve_scope("图书借阅或座位预约怎样获得第二课堂学分？", "current"),
            "current",
        )
        self.assertEqual(
            resolve_scope("烟雾报警器无故鸣叫或损坏怎么办？", "current"),
            "current",
        )

    def test_year_and_rule_specific_questions_use_historical_scope(self) -> None:
        historical_questions = (
            "2027 届毕业生生源信息确认怎么做？",
            "学校 2025 年国家助学金分几档，每档多少钱？",
            "国家奖学金和国家励志奖学金能不能同时获得，能不能同时申请国家助学金？",
            "家庭经济困难认定一年做几次，去年认定过今年还要重新申请吗？",
            "勤工助学一个人能同时在岗几个，怎么报名和录用？",
            "休学、保留学籍或退学后国家助学金还发吗？",
        )
        for question in historical_questions:
            with self.subTest(question=question):
                self.assertEqual(resolve_scope(question, "current"), "historical")

    def test_verified_current_gap_is_refused(self) -> None:
        self.assertTrue(is_known_current_gap("2026—2027 学年最终住宿费和入住房型是什么？"))
        self.assertTrue(is_known_current_gap("寝室几点熄灯？"))
        self.assertFalse(is_known_current_gap("按 2025 级公寓生活指南，寝室几点熄灯？"))


if __name__ == "__main__":
    unittest.main()
