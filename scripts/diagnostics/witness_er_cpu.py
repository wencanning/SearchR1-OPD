"""Seeded CPU forward witness; no pretrained model and no CUDA initialization."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import torch
import transformers
from transformers import Qwen2Config, Qwen2ForCausalLM
from verl.utils.evidence_residual import build_evidence_hidden_attention_mask

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.manual_seed(20260914)
config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
config._attn_implementation = 'eager'
model = Qwen2ForCausalLM(config).eval()
ids = torch.tensor([[1, 5, 9, 2, 11, 7]])
valid = torch.ones_like(ids)
zero = build_evidence_hidden_attention_mask(valid, torch.zeros_like(ids), torch.float32)
mask = build_evidence_hidden_attention_mask(valid, torch.tensor([[0, 0, 1, 1, 0, 0]]), torch.float32)
with torch.inference_mode():
    obs = model(ids, attention_mask=valid, use_cache=False).logits
    null = model(ids, attention_mask=zero, use_cache=False).logits
    hid = model(ids, attention_mask=mask, use_cache=False).logits
assert (obs - null).abs().max().item() < 1e-6
assert (obs[:, -1] - hid[:, -1]).abs().max().item() > 0
assert torch.isfinite(hid).all()
assert not torch.cuda.is_initialized()
assert all(p.device.type == 'cpu' for p in model.parameters())
print('WITNESS_CPU', json.dumps(dict(torch=torch.__version__, transformers=transformers.__version__,
    shape=list(obs.shape), device=str(obs.device), null_error=(obs-null).abs().max().item(),
    hidden_effect=(obs[:,-1]-hid[:,-1]).abs().max().item(), cuda_initialized=torch.cuda.is_initialized())))
