"""Read-only precision diagnostic for checkpoint75 vs saved step76 student logp."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='8'
os.environ['MKL_NUM_THREADS']='8'
import json
from contextlib import nullcontext
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/action-credit-recovery-20260914'
manifest=json.loads((OUT/'cpu_update_manifest.json').read_text())
torch.set_num_threads(8);torch.set_num_interop_threads(1)
model=AutoModelForCausalLM.from_pretrained(manifest['initial_checkpoint'],local_files_only=True,
    low_cpu_mem_usage=True,torch_dtype=torch.float32,attn_implementation='sdpa').eval()
tok=AutoTokenizer.from_pretrained(manifest['initial_checkpoint'],local_files_only=True)
for mode in ['fp32','autocast_bf16','weights_bf16']:
    if mode=='weights_bf16':model.to(torch.bfloat16)
    for c in manifest['cases'][:2]:
        ids=c['prompt']+c['ids'];n=len(c['prompt']);selected=[i for i,m in enumerate(c['mask']) if m]
        ctx=torch.autocast('cpu',dtype=torch.bfloat16) if mode=='autocast_bf16' else nullcontext()
        with torch.inference_mode(),ctx:
            h=model.model(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),
                position_ids=torch.arange(len(ids))[None],use_cache=False).last_hidden_state[0]
            values=[]
            for start in range(0,len(selected),64):
                inds=selected[start:start+64]
                logits=model.lm_head(h[torch.tensor([n+i-1 for i in inds])])[:,:len(tok)].float()
                values.extend(logits.log_softmax(-1).gather(-1,torch.tensor([c['ids'][i] for i in inds])[:,None]).squeeze(-1).tolist())
        errors=[abs(v-c['old'][i]) for v,i in zip(values,selected)]
        worst=sorted(zip(errors,selected,values),reverse=True)[:8]
        row=dict(mode=mode,key=c['key'],max_error=max(errors),mean_error=sum(errors)/len(errors),
            worst=[dict(error=e,index=i,token=tok.decode([c['ids'][i]]),old=c['old'][i],new=v,tag=c['tags'][i]) for e,i,v in worst])
        with (OUT/'student_precision_cpu.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)
print('PARITY_CHECK_COMPLETE',flush=True)
