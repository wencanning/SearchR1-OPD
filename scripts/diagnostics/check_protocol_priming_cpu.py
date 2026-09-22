"""Same-text, same-length initial-tag token-path intervention; CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='8'
os.environ['MKL_NUM_THREADS']='8'
import hashlib
import json
import time
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from probe_search_loop import ROOT,TEACHER,rows,events

OUT=ROOT/'reports/action-credit-recovery-20260914'
torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(42)
tok=AutoTokenizer.from_pretrained(ROOT/'data/student/0.5B',local_files_only=True)
data=pd.read_parquet(ROOT/'data/nq_hotpotqa_train_30k_no_cold_start/train.parquet')
lookup={(r['data_source'],int(r['extra_info']['index'])):r for r in data.to_dict('records')}
cases=[]
for method in ['opd','sod']:
    path,rs=rows(method,75)
    rs.sort(key=lambda r:hashlib.sha256(f'protocol-priming-v1|{method}|{r["index"]}'.encode()).hexdigest())
    selected=0
    for r in rs:
        ev=events(r)
        if not ev or r['token_ids'][:2]!=[13708,766]:continue
        source=lookup[(r['data_source'],int(r['index']))]
        prompt=tok.apply_chat_template(list(source['prompt']),add_generation_prompt=True,tokenize=True)
        original=prompt+r['token_ids'][:ev[0]['cut']]
        if len(original)>2048:continue
        changed=original.copy();changed[len(prompt):len(prompt)+2]=[27,26865]
        assert len(original)==len(changed) and tok.decode(original)==tok.decode(changed)
        cut=ev[0]['cut']
        cases.append(dict(key=f'{method}:75:{r["index"]}',question=source['question'],prompt=prompt,
            original=original,changed=changed,saved_initial_teacher_logp=r['teacher_log_prob'][0],
            saved_post_ids=r['token_ids'][cut:cut+3],saved_post_teacher_logps=r['teacher_log_prob'][cut:cut+3]))
        selected+=1
        if selected==4:break
manifest=dict(cases=cases,intervention='Only initial [13708,766] becomes [27,26865]; text and positions unchanged')
(OUT/'priming_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False))
model=AutoModelForCausalLM.from_pretrained(TEACHER,local_files_only=True,torch_dtype=torch.float32,
    low_cpu_mem_usage=True,attn_implementation='eager').eval()
assert next(model.parameters()).device.type=='cpu'
def next_lp(ids):
    with torch.inference_mode():
        logits=model(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),
            position_ids=torch.arange(len(ids))[None],use_cache=False,num_logits_to_keep=1).logits[0,-1,:len(tok)]
        return logits.log_softmax(-1)
def summarize(lp):
    top=lp.topk(5)
    return dict(top_ids=top.indices.tolist(),top_tokens=[tok.decode([i]) for i in top.indices.tolist()],top_logps=top.values.tolist(),
                canonical_first_logp=float(lp[13708]),split_first_logp=float(lp[27]))
for case in cases:
    start=time.time();opening=next_lp(case['prompt']);tokens=[];ids=case['prompt'].copy();lp=opening
    for _ in range(4):
        token=int(lp.argmax());tokens.append(token);ids.append(token)
        if '>' in tok.decode(tokens):break
        lp=next_lp(ids)
    original=next_lp(case['original']);changed=next_lp(case['changed'])
    row=dict(key=case['key'],initial=summarize(opening),initial_greedy_ids=tokens,initial_greedy_text=tok.decode(tokens),
        original_post=summarize(original),same_text_alt_path_post=summarize(changed),seconds=time.time()-start)
    with (OUT/'priming_cpu.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)
print('PRIMING_COMPLETE',flush=True)
