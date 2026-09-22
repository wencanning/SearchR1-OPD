"""CPU mathematical and protocol tests for the user-run GPU 2/3 experiment."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments/evidence_gap'))
from common import append, config, exact, frozen, normalize, read, save, train_command, training_source
from worker import Teacher, entropy, matched, observation, rollout, validate_annotation


class ExperimentTests(unittest.TestCase):
    def test_probe_interpreter_preserves_virtual_environment_path(self):
        self.assertEqual(config()['probe_python'],str(ROOT/'.venv/bin/python'))

    def test_smoke_dispatch_constructs_snapshot_jobs_without_gpu_execution(self):
        import launch
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch,MagicMock
        with tempfile.TemporaryDirectory() as tmp:
            c=config(); c['output']=tmp; source=Path(tmp)/'source'; source.mkdir()
            devices={2:{'uuid':'TEST-GPU2'},3:{'uuid':'TEST-GPU3'}}
            request=MagicMock(); request.json.return_value={'result':[]}
            executed=[]
            def fake_run(cmd,**kwargs):
                executed.append(cmd)
                self.assertEqual(kwargs['cwd'],source)
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'TEST-GPU2,TEST-GPU3')
                self.assertTrue(kwargs['env']['PYTHONPATH'].startswith(str(source)))
                opts=dict(x.split('=',1) for x in cmd[3:])
                save(Path(opts['trainer.default_local_dir'])/'actor/global_step_1/config.json',{})
            with patch.object(sys,'argv',['launch.py','smoke','--seeds','42']), \
                 patch('launch.config',return_value=c),patch('launch.prepare'), \
                 patch('launch.training_source',return_value=source), \
                 patch('launch.idle_or_fail',return_value=devices), \
                 patch('requests.post',return_value=request),patch('launch.subprocess.run',side_effect=fake_run), \
                 patch('launch.tempfile.mkdtemp',return_value=str(Path(tmp)/'ray')),redirect_stdout(io.StringIO()):
                launch.main()
            self.assertEqual(len(executed),3)
            with patch.object(sys,'argv',['launch.py','train','--dry-run']), \
                 patch('launch.config',return_value=c),patch('launch.training_source',side_effect=AssertionError('dry run mutated snapshot')), \
                 patch('launch.prepare',side_effect=AssertionError('dry run prepared data')),redirect_stdout(io.StringIO()):
                launch.main()

    def test_all_methods_keep_grpo_and_equal_budget(self):
        c=config(); opts={m:dict(x.split('=',1) for x in train_command(c,m,42)[3:]) for m in ('opd','sod','er')}
        for m,d in opts.items():
            self.assertEqual(d['algorithm.opd.grpo_reward_coef'],'1.0')
            self.assertEqual(d['actor_rollout_ref.rollout.n_agent'],'8')
            self.assertEqual(d['algorithm.opd.sod.allow_no_grpo_ablation'],'false')
            self.assertEqual(d['trainer.total_training_steps'],'151')
            self.assertEqual(d['trainer.n_gpus_per_node'],'2')
            self.assertEqual(d['data.train_data_source'],'hotpotqa')
        self.assertEqual(opts['sod']['algorithm.opd.sod.enable'],'true')
        self.assertEqual(opts['er']['algorithm.opd.teacher_target'],'evidence_residual')
        differing={k for k in opts['opd'] if len({d[k] for d in opts.values()})>1}
        self.assertEqual(differing,{'algorithm.opd.sod.enable','algorithm.opd.teacher_target','trainer.experiment_name','trainer.default_local_dir'})

    def test_blind_review_and_alias_gate(self):
        base=dict(key='x',reviewed=True,reviewer='A+B',sufficient=True,evidence_reason='Both hops shown',correct_aliases=['John Doe'],wrong_classes=[['Jane Doe']])
        validate_annotation(base)
        for patch in ({'reviewed':False},{'wrong_classes':[]},{'wrong_classes':[['John Doe']]},{'evidence_reason':''},
                      {'sufficient':False,'evidence_reason':''},{'wrong_classes':[['The John Doe']]}):
            with self.assertRaises(ValueError): validate_annotation(dict(base,**patch))
        self.assertTrue(exact('The Nile.',['Nile']))

    def test_frozen_resume_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'protocol.json'; frozen(p,{'x':1}); frozen(p,{'x':1})
            with self.assertRaises(ValueError): frozen(p,{'x':2})
            self.assertEqual(read(p),{'x':1})

    def test_prepared_panel_disjointness(self):
        c=config(); p=Path(c['output'])/'panels.json'
        if not p.exists(): self.skipTest('Run prepare first')
        panels=read(p); seen=set()
        for name,n in [('diagnostic',200),('evaluation',512),('development',128)]:
            qs={normalize(r['question']) for r in panels[name]}; self.assertEqual(len(qs),n)
            self.assertFalse(seen & qs); seen |= qs

    def test_entropy_matching_preserves_token_order_and_er_can_flip(self):
        import torch
        obs=torch.tensor([[1.,2.,0.],[-.5,1.,2.]])
        hidden=torch.tensor([[-2.,3.,0.],[-1.,.5,2.]])
        er=2*obs-hidden; z,gap=matched(obs,er)
        self.assertLess(gap,1e-4)
        self.assertTrue(torch.equal(z.argsort(-1),obs.argsort(-1)))
        self.assertNotEqual(int(er[0].argmax()),int(obs[0].argmax()))
        self.assertTrue(torch.allclose(entropy(z),entropy(er),atol=1e-4))

    def test_cached_hidden_attention_matches_uncached_on_tiny_qwen(self):
        import torch
        from transformers import Qwen2Config,Qwen2ForCausalLM
        torch.manual_seed(8)
        cfg=Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                       num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
        cfg._attn_implementation='sdpa'
        t=Teacher.__new__(Teacher); t.c={'teacher_dtype':'float32'}; t.torch=torch
        t.model=Qwen2ForCausalLM(cfg).eval(); t.vocab=32
        errs=t.witness([1,3,5,7,9],[0,0,1,1,0])
        self.assertLess(max(errs.values()),1e-5)
        a,_=t.forward([1,3,5,7,9],[0,0,1,1,0],True)
        b,_=t.forward([1,3,22,24,9],[0,0,1,1,0],True)
        # No intervening unmasked positions that could mediate the changed evidence.
        self.assertTrue(torch.allclose(a,b,atol=1e-6))
        observed,_=t.forward([1,3,5,7,9],[0,0,1,1,0])
        self.assertGreater(float((a-observed).abs().max()),1e-5)
        self.assertFalse(torch.cuda.is_initialized())

    def test_observation_retains_tags_and_only_marks_tool_body(self):
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(config()['student'],local_files_only=True)
        ids,mask=observation(tok,'A retrieved fact. '*100,40)
        text=tok.decode(ids)
        self.assertEqual(len(ids),40); self.assertEqual(len(ids),len(mask))
        self.assertIn('<information>',text); self.assertIn('</information>',text)
        self.assertGreater(sum(mask),0)
        visible=tok.decode([x for x,m in zip(ids,mask) if not m])
        self.assertIn('<information>',visible); self.assertIn('</information>',visible)

    def test_tool_rollout_preserves_native_answer_prefix(self):
        import torch
        from unittest.mock import patch
        from transformers import AutoTokenizer
        c=config(); tok=AutoTokenizer.from_pretrained(c['student'],local_files_only=True)
        class Model:
            device='cpu'
            turns=0
            def generate(self,x,**kw):
                action=['<think>Find the location.</think><search>capital France</search>',
                        '<think>The evidence gives Paris.</think><answer> Paris</answer>'][self.turns]
                self.turns+=1
                return torch.cat([x,torch.tensor([tok.encode(action,add_special_tokens=False)])],1)
        q=dict(prompt=[{'role':'user','content':'Capital of France?'}],gold=['Paris'])
        with patch('worker.retrieve',return_value='Paris is the capital of France.'):
            r=rollout(Model(),tok,c,q)
        self.assertTrue(r['reference_em']); self.assertEqual(r['searches'],1)
        self.assertIsNotNone(r['native']); self.assertTrue(any(r['native']['evidence_mask']))
        self.assertTrue(tok.decode(r['native']['prefix_ids']).endswith('<answer>'))
        self.assertNotIn('Paris</answer>',tok.decode(r['native']['prefix_ids']))

    def test_isolated_training_and_probe_share_intact_observation_boundaries(self):
        import importlib.util
        from types import SimpleNamespace
        from transformers import AutoTokenizer
        c=config(); source=training_source(c)
        name='search_r1.llm_agent.evidence_gap_snapshot_generation'
        spec=importlib.util.spec_from_file_location(name,source/'search_r1/llm_agent/generation.py')
        module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module)
        manager=module.LLMGenerationManager.__new__(module.LLMGenerationManager)
        manager.tokenizer=AutoTokenizer.from_pretrained(c['student'],local_files_only=True)
        for budget in (40,512):
            manager.config=SimpleNamespace(max_obs_length=budget)
            for suffix in ('.',',',':',')','plain'):
                body='A retrieved fact '*600+suffix
                ids,mask=manager._tokenize_observation_with_evidence_mask('\n\n<information>'+body+'</information>\n\n')
                text=manager.tokenizer.decode(ids)
                self.assertIn('</information>',text)
                self.assertEqual((ids,mask),observation(manager.tokenizer,body,budget))
        original=(ROOT/'search_r1/llm_agent/generation.py').read_text()
        self.assertIn('token_start < span_end and token_end > span_start',original)

    def test_summary_uses_all_eligible_states_and_keeps_c_pending(self):
        from summarize import summarize
        with tempfile.TemporaryDirectory() as tmp:
            c=config(); c['output']=tmp; c['bootstrap_samples']=50
            save(Path(tmp)/'panels.json',{'evaluation':[{'qid':'unused'}]})
            for i,(hidden,observed) in enumerate([(-2,-1),(-1,1),(0,-1)]):
                row=dict(key=str(i),qid=str(i),step=0,annotation={'sufficient':True},
                  margins={'hidden':hidden,'observed':observed,'er':1,'temperature':observed},
                  generations={t:dict(answer='x',stop_reason='answer_closed',reference_em=(i==0 if t=='er' else i==1),
                      reviewed_alias_em=(i==0 if t=='er' else i==1)) for t in ['hidden','observed','er','temperature']})
                append(Path(tmp)/'test'/'scores-0.jsonl',row)
                append(Path(tmp)/'test'/'candidates-0.jsonl',dict(key=str(i),step=0,answer='x',stop_reason='answer_closed'))
            summarize(c,'test'); d=read(Path(tmp)/'test'/'summary.json')
            for r in d['A']:
                if r['epsilon']==0:
                    self.assertEqual(r['eligible'],3); self.assertAlmostEqual(r['mean'],1/3)
            self.assertTrue(all(r['n']==1 for r in d['B']))
            self.assertEqual(d['C'],[]); self.assertEqual(len(d['C_pending']),9)
            self.assertEqual(d['denominators'][0]['complete_answers'],3)
            transitions={r['metric']:r for r in d['generation'] if r['target']=='er' and r['group']=='all_sufficient'}
            gain=transitions['Generated correction: wrong to correct']; harm=transitions['Generated harm: correct to wrong']
            self.assertEqual((gain['transitions'],gain['denominator']),(1,2))
            self.assertEqual((harm['transitions'],harm['denominator']),(1,1))
            self.assertEqual(transitions['Net generated correction vs observed']['mean'],0)

if __name__=='__main__': unittest.main()
