"""Post-run numeric audit only; does not change cases, interventions or decisions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from probe_independent_history import OUT,MODEL


def main():
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(42)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32,
        low_cpu_mem_usage=True,attn_implementation='eager').eval()
    assert next(model.parameters()).device.type=='cpu'
    cases=json.loads((OUT/'probe_manifest.json').read_text())['cases']
    original={r['key']:r for l in (OUT/'probe_scores.jsonl').read_text().splitlines() for r in [json.loads(l)]}
    results=[];start=time.time()
    for c in cases:
        # Every original wrong candidate + both candidates in the two near-tie T007 conditions.
        tasks=[('original','wrong')]
        if c['review_id']=='T007':tasks += [(condition,candidate) for condition in ['hide_claims','hide_all_post'] for candidate in ['gold','wrong']]
        for condition,candidate in tasks:
            labels=tok(c['leading']+c[candidate]+c['closing'],add_special_tokens=False)['input_ids']
            v=c['conditions'][condition];ids=v['ids']+labels[:-1];mask=v['mask']+[1]*(len(labels)-1)
            with torch.inference_mode():
                lp=model(input_ids=torch.tensor([ids]),attention_mask=torch.tensor([mask]),
                    position_ids=torch.arange(len(ids))[None,:],use_cache=False,num_logits_to_keep=len(labels)).logits[0].log_softmax(-1)
                chosen=lp.gather(-1,torch.tensor(labels)[:,None]).squeeze(-1)
            reference=original[c['key']]['scores'][condition][candidate]
            assert labels==reference['token_ids']
            row=dict(key=c['key'],review_id=c['review_id'],condition=condition,candidate=candidate,
                token_ids=labels,token_logps=chosen.tolist(),full_sum=chosen.sum().item(),
                max_cpu_gpu_error=max(abs(a-b) for a,b in zip(chosen.tolist(),reference['token_logps'])))
            results.append(row);print('CHECK',c['review_id'],condition,candidate,row['max_cpu_gpu_error'],flush=True)
    (OUT/'cpu_numeric_check.json').write_text(json.dumps(dict(results=results,seconds=time.time()-start),ensure_ascii=False,indent=2))
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
