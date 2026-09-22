"""Seeded CPU forward/backward/AdamW witness on a disposable 0.5B model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/action_calibration_cpu_20260916'
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20260916)
assert not torch.cuda.is_initialized()
path=ROOT/'verl_checkpoints/opd-grpo-0.5B/actor/global_step_50'
tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32,
                                        attn_implementation='eager',low_cpu_mem_usage=True).eval().to('cpu')
assert all(p.device.type=='cpu' and p.dtype==torch.float32 for p in model.parameters())
ids=tok.encode('Question: What color is snow?\n<answer>white</answer>',add_special_tokens=False)
x=torch.tensor([ids],device='cpu')
with torch.no_grad():
    before=model(x,use_cache=False).logits[:,:-1,:len(tok)].float().log_softmax(-1).gather(-1,x[:,1:,None]).squeeze(-1)
    cached=model(x,use_cache=True).logits[:,:-1,:len(tok)].float().log_softmax(-1).gather(-1,x[:,1:,None]).squeeze(-1)
assert float((before-cached).abs().max())<1e-4
opt=torch.optim.AdamW(model.parameters(),lr=1e-6,betas=(.9,.999),weight_decay=0.0)
lp=model(x,use_cache=False).logits[:,:-1,:len(tok)].float().log_softmax(-1).gather(-1,x[:,1:,None]).squeeze(-1)
loss=-lp.mean();loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
assert grad>0 and torch.isfinite(torch.tensor(grad));opt.step();opt.zero_grad(set_to_none=True)
with torch.no_grad():
    after=model(x,use_cache=False).logits[:,:-1,:len(tok)].float().log_softmax(-1).gather(-1,x[:,1:,None]).squeeze(-1)
change=float((after-before).abs().max());assert change>0 and torch.isfinite(after).all()
assert not torch.cuda.is_initialized()
OUT.mkdir(parents=True,exist_ok=True)
result=dict(passed=True,device='cpu',dtype='float32',threads=4,grad_norm=grad,max_logp_change=change,
            cache_parity_max_error=float((before-cached).abs().max()),cuda_initialized=False,no_checkpoint_written=True)
(OUT/'ENV_KERNEL_WITNESS.json').write_text(json.dumps(result,indent=2)+'\n')
print('CPU_BACKWARD_WITNESS_PASS',json.dumps(result),flush=True)
