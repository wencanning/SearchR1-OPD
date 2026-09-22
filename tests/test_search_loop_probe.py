"""CPU-only parser tests for frozen diagnostic contexts."""
import importlib.util
from pathlib import Path
import unittest

SPEC=importlib.util.spec_from_file_location('loop_probe',Path(__file__).resolve().parents[1]/'scripts/diagnostics/probe_search_loop.py')
probe=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(probe)

def record(parts):
    tokens=[];mask=[]
    for text,policy in parts:
        tokens.extend(text);mask.extend([policy]*len(text))
    return dict(token_texts=tokens,loss_mask=mask)

class ParsingTests(unittest.TestCase):
    def test_two_real_returns_and_unexecuted_request(self):
        r=record([('<search>q</search>',1),('<information>same</information>\n\n',0),
                  ('<search>q</search>',1),('<information>same</information>\n\n',0),
                  ('<search>q</search>',1)])
        ev=probe.events(r)
        self.assertEqual(len(ev),2)
        self.assertEqual(ev[0]['observation'],ev[1]['observation'])
        self.assertEqual(r['token_texts'][ev[1]['cut']],'<')
        self.assertTrue(r['loss_mask'][ev[1]['cut']])

    def test_model_generated_information_is_not_tool_result(self):
        r=record([('<search>q</search><information>fake</information><search>q</search>',1)])
        self.assertEqual(probe.events(r),[])

    def test_query_in_tool_text_is_not_action(self):
        r=record([('<search>q</search>',1),('<information>quoted <search>other</search></information>\n\n',0),
                  ('<answer>a</answer>',1)])
        ev=probe.events(r)
        self.assertEqual(len(ev),1)
        self.assertEqual(ev[0]['query'],'q')

    def test_query_normalization_is_only_case_and_whitespace(self):
        self.assertEqual(probe.norm(' Q  X\n'),probe.norm('q x'))
        self.assertNotEqual(probe.norm('q?'),probe.norm('q'))

if __name__=='__main__':unittest.main()
