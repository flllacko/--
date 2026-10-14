import unittest
from unittest.mock import patch
from app.qa.dialogue import plan_dialogue
from app.qa.generation import GenerationError


class DialogueTests(unittest.TestCase):
    def plan(self, raw, question="你是谁", private=False):
        with patch('app.qa.dialogue._call_ollama_chat', return_value=raw):
            return plan_dialogue(question, [], base_url='http://localhost', model='test', private=private)

    def test_general_answer(self):
        result = self.plan('{"mode":"general","reply":"我是学生服务助手"}')
        self.assertEqual(result['mode'], 'general')

    def test_known_gap_cannot_bypass_evidence(self):
        result = self.plan('{"mode":"general","reply":"23点","query":"现在寝室几点熄灯"}', '现在寝室几点熄灯')
        self.assertEqual(result['mode'], 'school')

    def test_private_scope_uses_retrieval(self):
        self.assertEqual(self.plan('{"mode":"general","reply":"内容"}', private=True)['mode'], 'school')

    def test_invalid_plan_does_not_become_answer(self):
        self.assertEqual(self.plan('{"mode":"general","reply":""}')['mode'], 'school')

    def test_malformed_plan_is_repaired(self):
        with patch('app.qa.dialogue._call_ollama_chat', side_effect=[
            '{"conclusion":"旧结构"}', '{"mode":"general","query":"你是谁","reply":"我是问答助手"}',
        ]) as caller:
            result = plan_dialogue('你是谁', [], base_url='http://localhost', model='test')
        self.assertEqual(result['mode'], 'general')
        self.assertEqual(caller.call_count, 2)
        self.assertIn('enum', caller.call_args.kwargs['response_schema']['properties']['mode'])

    def test_router_prompt_does_not_expand_unasked_details(self):
        with patch('app.qa.dialogue._call_ollama_chat', return_value='{"mode":"school","query":"我想办理休学","reply":""}') as caller:
            result = plan_dialogue('我想办休学', [], base_url='http://localhost', model='test')
        self.assertEqual(result['query'], '我想办休学')
        self.assertIn('不能擅自增加用户没有询问的材料、流程、时间、条件或结论', caller.call_args.args[0][0]['content'])

    def test_first_turn_does_not_accept_model_invented_facets(self):
        result = self.plan(
            '{"mode":"school","action":"answer",'
            '"query":"办理休学的流程、材料和时间节点是什么？",'
            '"reply":"","clarification":""}', '我想办休学',
        )
        self.assertEqual(result['query'], '我想办休学')

    def test_rewritten_follow_up_cannot_change_explicit_year(self):
        with patch('app.qa.dialogue._call_ollama_chat', return_value=(
            '{"mode":"school","action":"answer","depth":"specific",'
            '"facets":["time"],"query":"2026届就业手续什么时候办理？",'
            '"reply":"","clarification":""}'
        )):
            result = plan_dialogue(
                '2027届就业手续什么时候办理？',
                [{'role': 'user', 'content': '上一轮问过2026届'}],
                base_url='http://localhost', model='test',
            )
        self.assertEqual(result['query'], '2027届就业手续什么时候办理？')

    def test_clear_school_request_is_not_forced_to_clarify(self):
        result = self.plan(
            '{"mode":"general","action":"answer","depth":"overview","facets":[],"query":"我想办休学","reply":"","clarification":""}',
            '我想办休学',
        )
        self.assertEqual(result['mode'], 'school')
        self.assertEqual(result['action'], 'answer')
        self.assertEqual(result['depth'], 'overview')

    def test_model_can_request_one_targeted_clarification(self):
        result = self.plan(
            '{"mode":"general","action":"answer","query":"学校办理时间","reply":"",'
            '"clarification":"你是想问休学办理时间，还是复学办理时间？"}',
            '学校办理时间是什么时候？',
        )
        self.assertEqual(result['mode'], 'school')
        self.assertEqual(result['action'], 'clarify')
        self.assertIn('休学办理时间', result['clarification'])

    def test_known_evidence_gap_cannot_be_relabelled_as_ambiguity(self):
        result = self.plan(
            '{"mode":"school","action":"clarify","query":"寝室几点熄灯","reply":"",'
            '"clarification":"你想问哪栋宿舍？"}',
            '寝室几点熄灯？',
        )
        self.assertEqual(result['action'], 'answer')

    def test_explicit_step_question_cannot_be_collapsed_to_overview(self):
        result = self.plan(
            '{"mode":"school","action":"answer","depth":"overview",'
            '"facets":["steps"],"query":"2027届就业手续怎么办",'
            '"reply":"","clarification":""}',
            '2027届就业手续怎么办？',
        )
        self.assertEqual(result['depth'], 'specific')
