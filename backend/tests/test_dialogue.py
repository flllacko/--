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
        self.assertEqual(result['query'], '我想办理休学')
        self.assertIn('不能擅自增加用户没有询问的材料、流程、时间、条件或结论', caller.call_args.args[0][0]['content'])
