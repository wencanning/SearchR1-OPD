"""Independent CPU eager precision witness for frozen search-loop prefixes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='8'
os.environ['MKL_NUM_THREADS']='8'
import json
import time
import torch
from transformers import AutoModelForCausalLM
from probe_search_loop import OUT, TEACHER

torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(42)
manifest=json.loads((OUT/'manifest.json').read_text())
cases=[next(c for c in manifest['cases'] if c['method']==m and c['kind']==k)
       for m in ['opd','sod'] for k in ['repeat','control']]
model=AutoModelForCausalLM.from_pretrained(TEACHER,local_files_only=True,torch_dtype=torch.float32,
    low_cpu_mem_usage=True,attn_implementation='eager').eval()
assert next(model.parameters()).device.type=='cpu'
for c in cases:
    for state in c['states']:
        if state['name']=='hide_duplicate_observation':continue
        labels=state['saved_next_ids'];ids=state['ids']+labels[:-1];mask=state['mask']+[1]*(len(labels)-1)
        start=time.time()
        with torch.inference_mode():
            logits=model(input_ids=torch.tensor([ids]),attention_mask=torch.tensor([mask]),
                position_ids=torch.arange(len(ids))[None],use_cache=False,num_logits_to_keep=len(labels)).logits[0]
            values=logits.log_softmax(-1).gather(-1,torch.tensor(labels)[:,None]).squeeze(-1).tolist()
        row=dict(key=c['key'],state=state['name'],logps=values,seconds=time.time()-start,
            saved_fp16_max_error=max(abs(a-b) for a,b in zip(values,state['saved_teacher_logps'])))
        with (OUT/'cpu_eager.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('CPU_CHECK',row['key'],row['state'],row['saved_fp16_max_error'],row['seconds'],flush=True)
print('CPU_COMPLETE',flush=True)
