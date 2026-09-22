"""Teacher scoring of frozen student TRAIN continuations for a local OPD control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from probe import ROOT,GPU,TEACHER,STUDENT,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU and os.environ.get('RECOVERY_GPU_GUARDED')=='1'
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(24*1024**3/torch.cuda.get_device_properties(0).total_memory)
    source=ROOT/'reports/recovery_20260917_native/continue_student/continuations.jsonl';assert json.loads(source.with_name('status.json').read_text())['state']=='complete'
    rows=[r for r in map(json.loads,source.read_text().splitlines()) if r['split']=='train'];args.out.mkdir(exist_ok=True,parents=True)
    tok=AutoTokenizer.from_pretrained(TEACHER,local_files_only=True);assert tok.get_vocab()==AutoTokenizer.from_pretrained(STUDENT,local_files_only=True).get_vocab()
    mf=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),teacher=str(TEACHER),dtype='bfloat16',vocab=len(tok))
    path=args.out/'manifest.frozen.json'
    if path.exists():assert json.loads(path.read_text())==mf
    else:save(path,mf)
    model=AutoModelForCausalLM.from_pretrained(TEACHER,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True).eval().to('cuda')
    result=args.out/'scores.jsonl';old=[json.loads(l) for l in result.read_text().splitlines()] if result.exists() else [];done={(r['qid'],r['replicate']) for r in old}
    for r in rows:
        key=(r['qid'],r['replicate'])
        if key in done:continue
        ids=list(r['initial_ids']);positions=[];targets=[]
        for seg in r['segments']:
            for token,mask in zip(seg['ids'],seg['loss_mask']):
                if mask:positions.append(len(ids)-1);targets.append(token)
                ids.append(token)
        x=torch.tensor([ids[:-1]],device='cuda');lp=[]
        with torch.inference_mode():
            hidden=model.model(x,attention_mask=torch.ones_like(x),use_cache=False).last_hidden_state[0]
            for start in range(0,len(positions),32):
                z=model.lm_head(hidden[torch.tensor(positions[start:start+32],device='cuda')])[:,:len(tok)].float();ys=torch.tensor(targets[start:start+32],device='cuda')
                lp.extend(z.log_softmax(-1).gather(1,ys[:,None])[:,0].cpu().tolist())
        row=dict(qid=r['qid'],replicate=r['replicate'],positions=positions,targets=targets,teacher_log_probs=lp)
        with result.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        done.add(key);print('SCORE',len(done),len(rows),flush=True)
    save(args.out/'status.json',dict(state='complete',completed=len(done),total=len(rows)))


if __name__=='__main__':main()
