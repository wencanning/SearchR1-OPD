"""Seeded kernel and real local checkpoint witness, restricted to physical GPU 5."""
import json
import os

GPU = 'GPU-2abc6681-2116-879c-a361-749edb32ac3b'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(20260915)
torch.set_num_threads(2)
assert torch.cuda.device_count() == 1
x = torch.randn(32, 32, device='cuda', dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all()
path = 'data/student/0.5B'
tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    path, local_files_only=True, torch_dtype=torch.bfloat16,
    attn_implementation='sdpa', low_cpu_mem_usage=True).eval().to('cuda')
inputs = tok('The capital of France is', return_tensors='pt').to('cuda')
with torch.inference_mode():
    logits = model(**inputs).logits
    assert torch.isfinite(logits).all()
    out = model.generate(**inputs, do_sample=False, max_new_tokens=8,
                         pad_token_id=tok.eos_token_id)
assert out.shape[1] > inputs.input_ids.shape[1]
print('WITNESS_GPU5', json.dumps(dict(device=torch.cuda.get_device_name(),
    gpu_uuid=GPU, torch=torch.__version__, shape=list(logits.shape),
    generated=tok.decode(out[0]), finite=True)))
