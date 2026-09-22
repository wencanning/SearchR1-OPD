import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig

torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20260915)
tok=AutoTokenizer.from_pretrained('data/student/0.5B',local_files_only=True)
model=AutoModelForCausalLM.from_pretrained('data/student/0.5B',local_files_only=True,
    torch_dtype=torch.float32,attn_implementation='eager',low_cpu_mem_usage=True).eval().to('cpu')
x=tok('Two plus two equals',return_tensors='pt')
with torch.inference_mode():
    full=model(**x,use_cache=False).logits
    cached=model(**x,use_cache=True).logits
    assert torch.isfinite(full).all() and (full-cached).abs().max()<1e-5
    y=model.generate(**x,generation_config=GenerationConfig(do_sample=False,max_new_tokens=6,
        use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id))
assert y.shape[1]>x.input_ids.shape[1]
assert not torch.cuda.is_initialized() and all(p.device.type=='cpu' for p in model.parameters())
print('CPU_STOPPING_WITNESS',json.dumps(dict(device='cpu',dtype='float32',cuda_initialized=False,
    generated=tok.decode(y[0]),torch=torch.__version__,cache_error=float((full-cached).abs().max()))))
