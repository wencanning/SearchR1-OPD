"""Shared CPU-only experiment utilities; no HTTP clients or live training edits."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/action_calibration_cpu_20260916'
sp=importlib.util.spec_from_file_location('evidence_prefix_helpers',ROOT/'scripts/experiments/evidence_use/experiment.py')
M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M)
NEUTRAL='<think>I will use the available information to decide the next step.</think>\n'
CONDITIONS=['first_hop','both_hops','with_distractors']
TEACHER='/data/home/wencanning/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3'
BASE=ROOT/'verl_checkpoints/opd-grpo-0.5B/actor/global_step_50'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines()] if Path(path).exists() else []
def save(path,value):M.save(path,value)
def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush()
def rank(qid,salt):return hashlib.sha256((salt+'|'+qid).encode()).hexdigest()


def setup():
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20260916)
    assert not torch.cuda.is_initialized()
    return torch


def load_model(path):
    import torch
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32,
             attn_implementation='eager',low_cpu_mem_usage=True).eval().to('cpu')
    assert all(p.device.type=='cpu' for p in model.parameters())
    return model


def verify_inputs():
    manifest=M.read(OUT/'inputs.frozen.json')
    for path,h in manifest['source_sha256'].items():assert sha(ROOT/path)==h,('Frozen source changed',path)
    assert manifest['input_review_passed']
    return manifest


def path_score(model,ids,tag,vocab_size,torch):
    x=torch.tensor([ids+tag[:-1]],device='cpu')
    with torch.no_grad():
        z=model(x,attention_mask=torch.ones_like(x),use_cache=False,num_logits_to_keep=len(tag)).logits[0,:,:vocab_size].float()
        lp=z.log_softmax(-1)[torch.arange(len(tag)),torch.tensor(tag)]
    assert torch.isfinite(lp).all() and not torch.cuda.is_initialized()
    return dict(log_probability=float(lp.sum()),token_log_probabilities=lp.tolist())
