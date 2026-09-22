"""Complete the SAME frozen input panel for a CPU-FP32-only sensitivity view.

Primary history-reproduction exclusions remain untouched. This supplement does
not claim historical FP16 signal reproduction for excluded inputs.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
import json
import math
import shutil
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import probe_er_correction_cpu as base
from audit_pure_opd_mainline import norm

ROOT=base.ROOT
PRIMARY=base.OUT
OUT=PRIMARY/'precision_sensitivity'


def append(name,row):base.append_json(OUT/name,row)


def adaptive_matched_logits(obs,er):
    """Expand the label-free root bracket; retain the original 1e-4 gate."""
    target=base.entropy(er)
    centered=obs-obs.max(-1,keepdim=True).values
    lo=torch.full_like(target,math.log(.01));hi=torch.full_like(target,math.log(100.0))
    expansions=0
    for _ in range(12):
        lower=base.entropy(centered/lo.exp()[:,None])
        upper=base.entropy(centered/hi.exp()[:,None])
        low_bad=lower>target+1e-5;high_bad=upper<target-1e-5
        if not (low_bad.any() or high_bad.any()):break
        lo=torch.where(low_bad,lo-math.log(10),lo)
        hi=torch.where(high_bad,hi+math.log(10),hi);expansions+=1
    assert torch.all(base.entropy(centered/lo.exp()[:,None])<=target+1e-4), 'Unattainable target entropy (possible exact maximum ties)'
    assert torch.all(base.entropy(centered/hi.exp()[:,None])>=target-1e-4)
    for _ in range(35):
        mid=(lo+hi)/2;h=base.entropy(centered/mid.exp()[:,None])
        lo=torch.where(h<target,mid,lo);hi=torch.where(h<target,hi,mid)
    tau=((lo+hi)/2).exp();z=centered/tau[:,None]
    error=(base.entropy(z)-target).abs().max().item()
    assert error<1e-4,error
    return z,dict(temperatures=tau.tolist(),max_entropy_error=error,bracket_expansions=expansions)


def score_case(model,tok,c,boundary,alias=None):
    pref=c if boundary=='native' else c['early']
    answers={'original':c['answer']}
    if c['group']!='correct_control':answers['gold']=alias or c['gold_candidate']
    scores={};raw={};encoded={};checks={}
    for name,answer in answers.items():
        text=pref['leading']+answer+c['closing']
        e=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        labels=e['input_ids']
        content=[i for i,(a,b) in enumerate(e['offset_mapping']) if a<len(pref['leading']+answer) and b>len(pref['leading'])]
        ids=pref['prefix_ids']+labels[:-1];mask=pref['evidence_mask']+[0]*(len(labels)-1)
        obs=base.forward(model,ids,mask,len(labels))
        if boundary=='native' and name=='original':
            assert labels==c['original_labels']
            actual=obs.log_softmax(-1).gather(-1,torch.tensor(labels)[:,None]).squeeze(-1)
            error=(actual-torch.tensor(c['saved_teacher_logps'])).abs()
            checks.update(native_logp_max_error=error.max().item(),native_logp_mean_error=error.mean().item(),
                          historical_reproduction_pass=error.max().item()<.5)
        hid=base.forward(model,ids,mask,len(labels),True)
        d=OUT/'logits';d.mkdir(exist_ok=True)
        torch.save(dict(observed=obs,hidden=hid,labels=labels,content=content),
                   d/f"{c['review_id']}_{boundary}_{name}{'_alias' if alias else ''}.pt")
        zs,info=base.targets(obs,hid)
        scores[name]={m:base.stats(z,labels,content) for m,z in zs.items()}
        scores[name]['entropy_match_details']=info;raw[name]=zs;encoded[name]=labels
        print('SENS_SCORED',c['review_id'],boundary,name,alias,flush=True)
    if c['group']=='correct_control':scores['gold']=scores['original']
    divergence=None
    if c['group']!='correct_control':
        g,w=encoded['gold'],encoded['original'];k=next(i for i,(a,b) in enumerate(zip(g,w)) if a!=b)
        assert g[:k]==w[:k]
        err=max((raw['gold'][m][k]-raw['original'][m][k]).abs().max().item() for m in ['observed','hidden'])
        assert err<1e-3,err
        local={m:(z[k,g[k]]-z[k,w[k]]).item() for m,z in raw['gold'].items()}
        assert abs(local['er_1']-(2*local['observed']-local['hidden']))<1e-4
        divergence=dict(position=k,gold_token=tok.decode([g[k]]),wrong_token=tok.decode([w[k]]),log_odds=local,
                        shared_prefix_max_error=err,hidden_shared_prefix_max_error=err)
    return dict(excluded=False,scores=scores,divergence=divergence,checks=checks,
                scope='CPU FP32 teacher-internal sensitivity; historical FP16 parity not required')


def main():
    assert (PRIMARY/'COMPLETED.json').exists(), 'Run sequentially after the primary experiment'
    OUT.mkdir(exist_ok=True)
    for name in ['probe_manifest.json','sample_manifest.json','annotations.json']:
        if (OUT/name).exists():assert (OUT/name).read_bytes()==(PRIMARY/name).read_bytes()
        else:shutil.copyfile(PRIMARY/name,OUT/name)
    execution=dict(scope='same frozen CPU FP32 teacher; not historical FP16 signal reproduction',
        source_manifest_sha256=base.digest(PRIMARY/'probe_manifest.json'),
        primary_score_sha256=base.digest(PRIMARY/'scores.jsonl'),primary_completion=json.loads((PRIMARY/'COMPLETED.json').read_text()),
        amendment_sha256=base.digest(PRIMARY/'AMENDMENT_PRECISION_SENSITIVITY.md'),
        historical_gate_not_applied_here=True,alias_sensitivity={'E017':'Ernest Hemingway'},
        script_sha256=base.digest(Path(__file__)))
    base.write_frozen(OUT/'execution_manifest_v2.json',execution)
    base.matched_logits=adaptive_matched_logits
    manifest=json.loads((PRIMARY/'probe_manifest.json').read_text())
    cases=manifest['cases']
    primary={(r['key'],r['boundary']):r for l in (PRIMARY/'scores.jsonl').read_text().splitlines() for r in [json.loads(l)]}
    path=OUT/'scores.jsonl'
    done={(r['key'],r['boundary']):r for l in path.read_text().splitlines() for r in [json.loads(l)]} if path.exists() else {}
    for k,r in primary.items():
        if not r['excluded'] and k not in done:
            copied=dict(r,provenance='reused identical CPU FP32 result from primary',scope=execution['scope'])
            if r['boundary']=='native':copied['checks']['historical_reproduction_pass']=True
            append('scores.jsonl',copied);done[k]=copied
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(20260914)
    tok=AutoTokenizer.from_pretrained(base.MODEL,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(base.MODEL,local_files_only=True,torch_dtype=torch.float32,
        low_cpu_mem_usage=True,attn_implementation='eager').eval()
    model._probe_vocab_size=len(tok)
    assert len(tok)==151665 and all(p.device.type=='cpu' for p in model.parameters())
    for c in cases:
        for boundary in ['native']+(['early'] if c['early_probe'] else []):
            if (c['key'],boundary) in done:continue
            t=time.time();print('SENS_START',c['review_id'],boundary,flush=True)
            row=dict(key=c['key'],review_id=c['review_id'],group=c['group'],boundary=boundary,
                     **score_case(model,tok,c,boundary))
            row['seconds']=time.time()-t;append('scores.jsonl',row);done[(c['key'],boundary)]=row
            print('SENS_DONE',c['review_id'],boundary,row['seconds'],flush=True)
    if not (OUT/'alias_sensitivity.json').exists():
        c=next(c for c in cases if c['review_id']=='E017')
        row=score_case(model,tok,c,'native',alias='Ernest Hemingway')
        row.update(review_id='E017',alias='Ernest Hemingway',post_hoc=True)
        (OUT/'alias_sensitivity.json').write_text(json.dumps(row,ensure_ascii=False,indent=2))
    gen_path=OUT/'generation.jsonl'
    generated={(r['key'],r['target']) for l in gen_path.read_text().splitlines() for r in [json.loads(l)]} if gen_path.exists() else set()
    for l in (PRIMARY/'generation.jsonl').read_text().splitlines():
        r=json.loads(l)
        if (r['key'],r['target']) not in generated:
            append('generation.jsonl',dict(r,provenance='reused primary FP32 continuation'))
            generated.add((r['key'],r['target']))
    for c in cases:
        if not c['free_generate']:continue
        for target in manifest['generation_targets']:
            if (c['key'],target) in generated:continue
            new=[];t=time.time();reason='max_tokens'
            for i in range(manifest['generation_max_tokens']):
                ids=c['prefix_ids']+new;mask=c['evidence_mask']+[0]*len(new)
                obs=base.forward(model,ids,mask,1)
                z=obs if target=='observed' else 2*obs-base.forward(model,ids,mask,1,True)
                if i==0:assert int(z[0].argmax())==done[(c['key'],'native')]['scores']['original'][target]['top_ids'][0]
                new.append(int(z[0].argmax()));text=tok.decode(new,skip_special_tokens=False)
                print('SENS_GEN',c['review_id'],target,i+1,repr(text),flush=True)
                if '</answer>' in text:reason='answer_closed';break
                if new[-1]==tok.eos_token_id:reason='eos';break
            text=tok.decode(new,skip_special_tokens=False)
            answer=text.split('</answer>')[0].strip() if reason=='answer_closed' else None
            append('generation.jsonl',dict(key=c['key'],review_id=c['review_id'],group=c['group'],target=target,
                ids=new,text=text,answer=answer,stop_reason=reason,
                reference_em=norm(answer) in [norm(g) for g in c['gold']] if answer is not None else False,
                historical_reproduction_pass=done[(c['key'],'native')]['checks']['historical_reproduction_pass'],
                seconds=time.time()-t))
    (OUT/'COMPLETED.json').write_text(json.dumps(dict(completed=True,scope=execution['scope'],device='cpu',
        cuda_initialized=torch.cuda.is_initialized(),score_rows=len(done),
        historical_failures=[c['review_id'] for c in cases if not done[(c['key'],'native')]['checks']['historical_reproduction_pass']]),indent=2))
    print('SENSITIVITY_COMPLETE_CPU',flush=True)


if __name__=='__main__':main()
