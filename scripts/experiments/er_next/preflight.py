"""CPU-only meaningful preflight; synthetic fixtures are tests, never results."""
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[3]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from collect_forks import parse_forced
    from analyze_forks import crossfit_value
    import run_gpu5_guarded as guard
    from unittest.mock import patch, Mock
    with patch.object(guard, 'memory', return_value=100), patch.object(guard.subprocess, 'Popen') as launch, patch.object(guard.sys, 'argv', ['guard', '--states', 'test']):
        try:
            guard.main()
        except SystemExit:
            pass
        else:
            raise AssertionError('Low memory must prevent launch')
        launch.assert_not_called()
    worker=Mock(pid=123456); worker.poll.return_value=None; worker.wait.return_value=-15
    with patch.object(guard, 'memory', side_effect=[25*1024, 11*1024]), patch.object(guard.subprocess, 'Popen', return_value=worker), patch.object(guard.os, 'killpg') as kill, patch.object(guard.sys, 'argv', ['guard', '--states', 'test']):
        try:
            guard.main()
        except RuntimeError:
            pass
        else:
            raise AssertionError('Free memory guard must terminate own worker')
        kill.assert_called_once_with(worker.pid, guard.signal.SIGTERM)
    assert parse_forced('Paris</answer>', 'answer')=='Paris'
    assert parse_forced('Paris', 'answer') is None
    assert parse_forced('<search>Paris</search></answer>', 'answer') is None
    assert parse_forced('x</answer>garbage', 'answer') is None
    assert parse_forced('</answer>', 'answer') is None
    # Balanced artificial feature -> action utility; test fold boundaries and no result leakage.
    data={}; ids=[f'test{i}' for i in range(48)]
    for i,s in enumerate(ids):
        for b in ('answer','search'):
            for r in range(4):
                data[s,b,r]=dict(qid=s,em=(b=='search')==(i%2==1),
                    preference=dict(logp_answer=-1.,logp_search=-1.,hidden_state=[float(i%2)]))
    tested=crossfit_value(dict(state_ids=ids,replicates=4,exploratory=True),data)
    assert tested['descriptive_positive_intervals'] and not tested['authorizes_training']
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    start=time.monotonic()
    path=ROOT/'verl_checkpoints/opd-grpo-0.5B/actor/global_step_50'
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32,
        attn_implementation='eager',low_cpu_mem_usage=True).eval().to('cpu')
    states=json.loads((ROOT/'reports/er_next_20260916/fork_states.frozen.json').read_text())['states']
    s=states[0]
    assert tok.decode(s['prefix_ids'],skip_special_tokens=False)==s['prefix_text']
    x=torch.tensor([s['prefix_ids']])
    with torch.inference_mode():
        h=model.model(x,attention_mask=torch.ones_like(x),use_cache=False).last_hidden_state[0,-1]
    assert torch.isfinite(h).all()
    scores={}
    for name in ('answer','search'):
        tag=tok.encode('<'+name+'>',add_special_tokens=False)
        z=torch.tensor([s['prefix_ids']+tag[:-1]])
        with torch.inference_mode():
            logits=model(z,attention_mask=torch.ones_like(z),use_cache=False,num_logits_to_keep=len(tag)).logits[0,:,:len(tok)].float()
        assert logits.shape==(len(tag),len(tok)) and torch.isfinite(logits).all()
        scores[name]=float(logits.log_softmax(-1)[torch.arange(len(tag)),torch.tensor(tag)].sum())
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',device='cpu',cuda_initialized=False,qid=s['qid'],hidden_width=len(h),
        canonical_path_logp=scores,seconds=time.monotonic()-start,
        checks=['mocked low-memory launch refusal','mocked watchdog stops only owned process group',
                'malformed forced output rejection','crossfit fixture direction','exploratory gate cannot authorize training',
                'native token prefix parity','real 0.5B CPU hidden state','real canonical path scoring'],
        limitations=['No complete real retrieval branch executed','GPU runtime guard not exercised with CUDA','No training executed'])
    (ROOT/'reports/er_next_20260916/PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
