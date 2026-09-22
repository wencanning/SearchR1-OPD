"""Real CPU scoring/gradient checks and temporary synthetic statistic fixtures."""
import contextlib
import gc
import io
import json
import tempfile
from pathlib import Path
import numpy as np
from common import ROOT,OUT,BASE,TEACHER,M,setup,load_model,verify_inputs,path_score,sha,save


def main():
    import transformers
    from transformers import AutoTokenizer,GenerationConfig,LogitsProcessor
    from verl.trainer.ppo.core_algos import compute_policy_loss
    from local_update import roles
    import analyze_bias as B
    torch=setup();m=verify_inputs();tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    assert len(tok)==m['vocab_size']
    groups=[set(v) for v in m['split_qids'].values()]
    assert len(set.union(*groups))==56 and all(not(a&b) for i,a in enumerate(groups) for b in groups[i+1:])
    for c in m['cases']:assert len(c['ids'])==len(c['evidence_mask']) and sum(c['evidence_mask'])>0
    role_ids=tok.encode('<answer>Paris</answer>\n<search>Paris population</search>',add_special_tokens=False)
    rr=roles(tok,role_ids);assert sum(r in ['action','action_mixed'] for r in rr)==2
    assert all(r in ['action','action_mixed','format','body'] for r in rr) and 'body' in rr and 'format' in rr
    # Test b recovery and paired comparison arithmetic; all synthetic data is temporary.
    original_out,original_verify=B.OUT,B.verify_inputs
    with tempfile.TemporaryDirectory(prefix='action_bias_selfcheck_') as name:
        B.OUT=Path(name);B.verify_inputs=lambda:m
        (B.OUT/'inputs.frozen.json').write_text(json.dumps(m))
        for split in ['calibration','test']:
            data=[]
            for cp in m['checkpoints']:
                for i,c in enumerate(x for x in m['cases'] if x['split']==split):
                    value=i/100+{'first_hop':-2,'both_hops':1,'with_distractors':-1.5}[c['condition']]+(1.25 if cp['method']=='er' else 0)
                    data.append(dict(label=cp['label'],qid=c['qid'],condition=c['condition'],log_odds=value,device='cpu',cuda_initialized=False))
            (B.OUT/f'scores_{split}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in data))
        with contextlib.redirect_stdout(io.StringIO()):B.calibrate();B.summarize()
        assert all(abs(v['b']-1.25)<1e-8 for v in M.read(B.OUT/'bias.frozen.json')['biases'].values())
        assert all(abs(v['delta'])<1e-8 for v in M.read(B.OUT/'bias_paired.json') if 'system' not in v)
    B.OUT, B.verify_inputs=original_out,original_verify
    model=load_model(BASE);c=next(c for c in m['cases'] if c['split']=='calibration')
    scores={a:path_score(model,c['ids'],tag,m['vocab_size'],torch) for a,tag in m['tags'].items()}
    common_error=abs(scores['answer']['token_log_probabilities'][0]-scores['search']['token_log_probabilities'][0]);assert common_error<1e-3
    class Vocab(LogitsProcessor):
        def __call__(self,ids,z):z[:,m['vocab_size']:]=-float('inf');return z
    x=torch.tensor([c['ids']],device='cpu');config=GenerationConfig(do_sample=True,temperature=1.,top_k=0,top_p=1.,max_new_tokens=3,
        use_cache=True,eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id,return_dict_in_generate=True,output_scores=True)
    with torch.no_grad():output=model.generate(x,attention_mask=torch.ones_like(x),generation_config=config,logits_processor=[Vocab()])
    ys=output.sequences[0,len(c['ids']):].tolist();assert ys
    saved=torch.tensor([float(z[0,:m['vocab_size']].float().log_softmax(-1)[y]) for z,y in zip(output.scores,ys)])
    x=torch.tensor([c['ids']+ys[:-1]],device='cpu')
    z=model(x,attention_mask=torch.ones_like(x),use_cache=False,num_logits_to_keep=len(ys)).logits[0,:,:m['vocab_size']].float()
    lp=z.log_softmax(-1)[torch.arange(len(ys)),torch.tensor(ys)]
    error=float((lp.detach()-saved).abs().max());assert error<1e-3
    advantages=torch.tensor([[.01*(-1)**i for i in range(len(ys))]])
    loss,*_=compute_policy_loss(lp.detach()[None,:],lp[None,:],advantages,torch.ones_like(lp)[None,:],.2,.2,.28,3.)
    loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert norm>0 and np.isfinite(norm)
    assert not torch.cuda.is_initialized();model.zero_grad(set_to_none=True)
    checks=dict(passed=True,split_sizes={k:len(v) for k,v in m['split_qids'].items()},test_update_disjoint=True,
        bias_statistic_fixture_pass=True,role_parser_pass=True,common_tag_probability_error=common_error,
        generation_no_cache_logp_error=error,real_ppo_gradient_norm=norm,device='cpu',dtype='float32',threads=4,
        cuda_initialized=False,torch_version=torch.__version__,transformers_version=transformers.__version__,
        no_checkpoint_written=True)
    save(OUT/'PRELAUNCH_CHECKS.json',checks)
    sources=list((ROOT/'scripts/experiments/action_calibration_cpu').glob('*.py'))+[ROOT/'scripts/experiments/action_calibration_cpu/run_cpu.sh',
        ROOT/'verl/utils/evidence_residual.py',ROOT/'verl/trainer/ppo/core_algos.py']
    paths=[ROOT/cp['path'] for cp in m['checkpoints']]+[Path(TEACHER)]
    snapshots={}
    for folder in paths:
        for path in folder.iterdir():
            if path.is_file() and (path.suffix in ['.safetensors','.json','.txt']):
                key=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
                stat=path.stat();snapshots[key]=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    execution=dict(code_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},checkpoint_files=snapshots,
                   input_sha256=sha(OUT/'inputs.frozen.json'),environment='action-calibration-cpu@ab843930')
    target=OUT/'execution.frozen.json'
    if target.exists():assert M.read(target)==execution
    else:save(target,execution)
    print('ACTION_CALIBRATION_PREFLIGHT_PASS',json.dumps(checks),flush=True)


if __name__=='__main__':main()
