"""Bounded, CPU-only frozen-prefix candidate likelihood probe (no training)."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import copy
import gzip
import hashlib
import json
import re
import time
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.utils.evidence_residual import build_evidence_hidden_attention_mask

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reports/offline_er_cpu_v3'
MODEL = '/data/home/wencanning/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3'


def select():
    cases=json.loads((ROOT/'reports/paired_coef1_first20/review_sample.json').read_text())
    # Case1 has malformed protocol closure. Exclude instead of guessing its mask.
    chosen=[dict(c, group='audited_failure') for c in cases if c['review_id'] in [3,7,9,10,11]]
    rows=[json.loads(l) for l in (ROOT/'reports/paired_coef1_first20/trajectories.jsonl').read_text().splitlines()]
    paired={(r['method'],r['global_step'],r['index']):r for r in rows}
    used={c['index'] for c in chosen}
    controls=[]
    for r in rows:
        other=paired[('sod' if r['method']=='opd' else 'opd',r['global_step'],r['index'])]
        if r['sequence_score']==1 and r['bounded_gold_hit'] and other['sequence_score']==0 and other['final_answer'] and r['index'] not in used:
            if any(g.lower() in other['final_answer'].lower() for g in r['gold']):continue
            controls.append(dict(r, group='EM_correct_control', competitor=other['final_answer']))
    controls.sort(key=lambda r:hashlib.sha256(f"probe42|{r['question_id']}".encode()).hexdigest())
    for r in controls:
        if r['index'] in used:continue
        chosen.append(r);used.add(r['index'])
        if len(chosen)==10:break
    return chosen


def saved_record(c):
    with gzip.open(ROOT/c['source_file'],'rt') as f:
        next(f)
        return next(r for r in map(json.loads,f) if r['index']==c['index'])


def prepare(c, tok, source):
    r=saved_record(c)
    positions=[i for i,s in enumerate(r['segments']) if s=='answer' and r['loss_mask'][i]]
    assert positions
    end=positions[0]
    text=''.join(r['token_texts'][:end])
    assert text.count('<information>')==text.count('</information>'), 'Malformed information tags'
    spans=[(m.start(1),m.end(1)) for m in re.finditer(r'<information>(.*?)</information>',text,re.S)]
    offset=0; evidence=[]
    for i,t in enumerate(r['token_texts'][:end]):
        stop=offset+len(t)
        evidence.append(int(not r['loss_mask'][i] and any(offset<b and stop>a for a,b in spans)))
        offset=stop
    prompt=tok.apply_chat_template(list(source['prompt']),add_generation_prompt=True,tokenize=False)
    prompt_ids=tok(prompt,add_special_tokens=False)['input_ids']
    ids=prompt_ids+r['token_ids'][:end]
    assert sum(evidence)>0
    # Identical saved policy prefix for all candidates and both teacher views.
    return ids,[0]*len(prompt_ids)+evidence


@torch.inference_mode()
def prefix_forward(model,ids,evidence,hidden):
    x=torch.tensor([ids]);valid=torch.ones_like(x)
    mask=build_evidence_hidden_attention_mask(valid,torch.tensor([evidence]),torch.float32) if hidden else valid
    out=model.model(input_ids=x,attention_mask=mask,position_ids=torch.arange(len(ids))[None,:],use_cache=True)
    logits=model.lm_head(out.last_hidden_state[:,-1:,:]).squeeze(0)
    return logits,out.past_key_values


@torch.inference_mode()
def candidate_logits(model,prefix_result,ids,evidence,tokens,hidden):
    # Full forward: the initial cached implementation FAILED a numerical parity
    # test (max logit error 38.36); never use its scores as research evidence.
    x=torch.tensor([ids+tokens[:-1]])
    valid=torch.ones_like(x)
    mask=(build_evidence_hidden_attention_mask(valid,
          torch.tensor([evidence+[0]*(len(tokens)-1)]),torch.float32) if hidden else valid)
    return model(input_ids=x,attention_mask=mask,
        position_ids=torch.arange(x.shape[1])[None,:],use_cache=False,
        num_logits_to_keep=len(tokens)).logits.squeeze(0)


def score(z,labels):
    assert torch.isfinite(z).all()
    lp=z.log_softmax(-1)
    value=lp.gather(-1,labels[:,None]).squeeze(-1)
    return {'sum':value.sum().item(),'mean':value.mean().item(),
            'entropy':(-(lp.exp()*lp).sum(-1)).mean().item()}


def main():
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(42)
    OUT.mkdir(parents=True,exist_ok=True)
    chosen=select()
    (OUT/'manifest.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2))
    print('CPU ONLY; threads=8; FP32; cases=',len(chosen),flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    df=pd.read_parquet(ROOT/'data/nq_hotpotqa_train_30k_no_cold_start/train.parquet')
    lookup={(r['data_source'],r['extra_info']['index']):r for r in df.to_dict('records')}
    started=time.time()
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32,
        attn_implementation='eager',local_files_only=True,low_cpu_mem_usage=True).eval()
    assert next(model.parameters()).device.type=='cpu'
    print('Loaded in',time.time()-started,flush=True)
    results=OUT/'scores.jsonl'
    done=set()
    if results.exists():
        done={r['key'] for r in map(json.loads,results.read_text().splitlines())}
    if (OUT/'excluded.jsonl').exists():
        done.update(r['key'] for r in map(json.loads,(OUT/'excluded.jsonl').read_text().splitlines()))
    for ordinal,c in enumerate(chosen,1):
        key=f"{c['method']}:{c['global_step']}:{c['index']}"
        if key in done:continue
        if c['index']==49679:
            with (OUT/'excluded.jsonl').open('a') as f:
                f.write(json.dumps({'key':key,'reason':'Ambiguous nationality/ancestry granularity; excluded before v3 scoring'})+'\n')
            continue
        started=time.time()
        try:
            ids,evidence=prepare(c,tok,lookup[(c['data_source'],c['index'])])
        except AssertionError as exc:
            # Protocol exclusions are not silently replaced with easier examples.
            with (OUT/'excluded.jsonl').open('a') as f:
                f.write(json.dumps({'key':key,'reason':str(exc)})+'\n')
            print('EXCLUDED',key,str(exc),flush=True)
            continue
        print('CASE',ordinal,key,'prefix',len(ids),'evidence',sum(evidence),flush=True)
        observed=hidden=None
        candidates={'gold':c['gold'][0],'competitor':c.get('competitor',c['final_answer'])}
        output=dict(key=key,group=c['group'],question=c['question'],review_id=c.get('review_id'),
                    candidates=candidates,prefix_tokens=len(ids),evidence_tokens=sum(evidence),scores={})
        for name,answer in candidates.items():
            # Saved prefix already includes the newline following <answer>.
            tokens=tok(answer.strip()+'\n</answer>',add_special_tokens=False)['input_ids']
            labels=torch.tensor(tokens)
            obs=candidate_logits(model,observed,ids,evidence,tokens,False)
            if (c['group']=='audited_failure' and name=='competitor') or (c['group']=='EM_correct_control' and name=='gold'):
                record=saved_record(c)
                start=next(i for i,s in enumerate(record['segments']) if s=='answer' and record['loss_mask'][i])
                if tokens[0]==record['token_ids'][start]:
                    actual=obs[0].log_softmax(-1)[tokens[0]].item()
                    reference=record['teacher_log_prob'][start]
                    output['first_token_reproduction']={'cpu_fp32':actual,'saved_gpu_fp16':reference,'abs_error':abs(actual-reference)}
                    if abs(actual-reference)>=.5:
                        with (OUT/'excluded.jsonl').open('a') as f:
                            f.write(json.dumps({'key':key,'reason':'CPU/GPU first-token baseline error >= .5 nat',
                                               'check':output['first_token_reproduction']})+'\n')
                        print('EXCLUDED baseline mismatch',key,output['first_token_reproduction'],flush=True)
                        break
            print(' ',name,'observed_done_s=',round(time.time()-started,2),flush=True)
            hid=candidate_logits(model,hidden,ids,evidence,tokens,True)
            if ordinal==1 and name=='gold':
                null=candidate_logits(model,None,ids,[0]*len(ids),tokens,True)
                err=(obs-null).abs().max().item()
                assert err<1e-3,err
                output['zero_evidence_max_logit_error']=err
                print(' no-evidence mask check max_logit_error=',err,flush=True)
            specs={'observed':obs,'hidden':hid,'er_0.5':obs+.5*(obs-hid),
                   'er_1':2*obs-hid,'er_1.5':2.5*obs-1.5*hid,'temperature_0.7':obs/.7,'temperature_0.5':obs/.5}
            values={k:score(v,labels) for k,v in specs.items()}
            values['tokens']=len(tokens)
            assert torch.equal(obs+0*(obs-hid),obs)
            output['scores'][name]=values
        if len(output['scores'])!=2:continue
        output['seconds']=time.time()-started
        with results.open('a') as f:f.write(json.dumps(output,ensure_ascii=False)+'\n')
        print('DONE',ordinal,'seconds',round(output['seconds'],2),
              'margins',{k:round(output['scores']['gold'][k]['sum']-output['scores']['competitor'][k]['sum'],3)
                         for k in specs},flush=True)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
