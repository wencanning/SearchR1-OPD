"""Independent frozen-teacher ER mechanism test, CPU FP32 only.

Teacher-only likelihood/continuation diagnostic, not a training comparison.
Full uncached forwards intentionally avoid previously invalidated cache paths.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
import argparse
import gzip
import hashlib
import json
import math
import re
import time
from pathlib import Path

import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.utils.evidence_residual import build_evidence_hidden_attention_mask
from audit_pure_opd_mainline import norm
from sample_er_correction_cpu import ROOT, OUT, SEED, sha

MODEL = '/data/home/wencanning/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_frozen(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Frozen manifest changed: {path}'
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def prepare(tok):
    sample = json.loads((OUT / 'sample_manifest.json').read_text())
    annotations = {r['review_id']: r for r in json.loads((OUT / 'annotations.json').read_text())['labels']}
    data = pd.read_parquet(ROOT / 'data/nq_hotpotqa_train_30k_no_cold_start/train.parquet')
    lookup = {(r['data_source'], int(r['extra_info']['index'])): r for r in data.to_dict('records')}
    groups = {'sufficient_failure': [], 'correct_control': [], 'insufficient_control': []}
    for c in sample['cases']:
        a = annotations[c['review_id']]
        if a['semantic'] == 'wrong' and a['evidence'] == 'sufficient':
            group = 'sufficient_failure'
        elif c['score'] == 1 and a['semantic'] == 'correct' and a['evidence'] == 'sufficient':
            group = 'correct_control'
        elif a['semantic'] == 'wrong' and a['evidence'] in ['absent', 'partial']:
            group = 'insufficient_control'
        else:
            continue
        groups[group].append(dict(c, group=group, annotation=a))
    exclusions, cases = [], []
    # Freeze technical exclusions before model scores, then deterministic caps.
    prepared = {k: [] for k in groups}
    for group, selected in groups.items():
        for c in sorted(selected, key=lambda c: sha(SEED + '|probe|' + c['key'])):
            try:
                assert c['answer'], 'No final answer'
                assert not c['malformed_information'], 'Malformed information tags'
                source_path = ROOT / c['source_file']
                assert digest(source_path) == sample['source_hashes'][c['source_file']]
                with gzip.open(source_path, 'rt') as f:
                    next(f)
                    r = next(r for l in f for r in [json.loads(l)] if int(r['index']) == c['index'])
                response = ''.join(r['token_texts'])
                assert response == c['response']
                am = list(re.finditer(r'<answer>(.*?)</answer>', response, re.S))[-1]
                start, end = am.span(1)
                while response[start].isspace(): start += 1
                while response[end-1].isspace(): end -= 1
                bounds, pos = [], 0
                for t in r['token_texts']:
                    bounds.append((pos, pos+len(t))); pos += len(t)
                cut = next(i for i, (a, b) in enumerate(bounds) if a <= start < b)
                lead = response[bounds[cut][0]:start]
                assert not any(ch in lead for ch in '<>'), 'Answer boundary overlaps tag'
                closing = response[end:am.end()]
                native_labels = tok(lead+c['answer']+closing, add_special_tokens=False)['input_ids']
                assert native_labels == r['token_ids'][cut:cut+len(native_labels)], 'Native token boundary mismatch'
                assert all(r['loss_mask'][i] for i in range(cut, cut+len(native_labels))), 'Non-policy answer'
                source = lookup[(c['data_source'], c['index'])]
                prompt = tok.apply_chat_template(list(source['prompt']), add_generation_prompt=True, tokenize=True)
                spans = [m.span(1) for m in re.finditer(r'<information>(.*?)</information>', response[:bounds[cut][0]], re.S)]
                evidence = [int(not r['loss_mask'][i] and any(a < hi and b > lo for lo, hi in spans))
                            for i, (a, b) in enumerate(bounds[:cut])]
                assert sum(evidence) > 0
                assert all(not r['loss_mask'][i] for i, e in enumerate(evidence) if e)
                c.update(leading=lead, closing=closing, prefix_ids=prompt+r['token_ids'][:cut],
                         evidence_mask=[0]*len(prompt)+evidence,
                         original_labels=native_labels,
                         saved_teacher_logps=r['teacher_log_prob'][cut:cut+len(native_labels)],
                         saved_student_logps=r['student_log_prob'][cut:cut+len(native_labels)],
                         saved_opd_advantage=r['opd_advantage'][cut:cut+len(native_labels)],
                         saved_weighted_advantage=r['weighted_opd_advantage'][cut:cut+len(native_labels)])
                # Secondary boundary: remove only the final post-observation
                # response and explicitly cue an answer. This is not on-policy.
                last_info = list(re.finditer(r'<information>.*?</information>', response[:bounds[cut][0]], re.S))[-1]
                boundary = next(i+1 for i, (_, b) in enumerate(bounds) if b >= last_info.end())
                assert boundary <= cut
                assert response[last_info.end():bounds[boundary-1][1]].strip() == ''
                cue = tok('\n\n<answer>\n', add_special_tokens=False)['input_ids']
                c['early'] = dict(prefix_ids=prompt+r['token_ids'][:boundary]+cue,
                                  evidence_mask=[0]*len(prompt)+evidence[:boundary]+[0]*len(cue),
                                  leading='')
                prepared[group].append(c)
            except (AssertionError, StopIteration) as exc:
                exclusions.append(dict(review_id=c['review_id'], key=c['key'], group=group, reason=str(exc)))
    for group, cap in [('sufficient_failure', 12), ('correct_control', 4), ('insufficient_control', 2)]:
        cases.extend(prepared[group][:cap])
    failures = [c for c in cases if c['group'] == 'sufficient_failure']
    controls = [c for c in cases if c['group'] == 'correct_control']
    early_keys = {c['key'] for c in failures[:2]}
    gen_keys = {c['key'] for c in failures[:2]+controls[:1]}
    for c in cases:
        c['early_probe'] = c['key'] in early_keys
        c['free_generate'] = c['key'] in gen_keys
        c['gold_candidate'] = c['answer'] if c['group'] == 'correct_control' else c['gold'][0]
    manifest = dict(seed=SEED, model=MODEL, dtype='float32', device='cpu', attention='eager', use_cache=False,
                    threads=8, torch=torch.__version__, transformers=transformers.__version__,
                    sample_sha256=digest(OUT/'sample_manifest.json'), annotations_sha256=digest(OUT/'annotations.json'),
                    primary_alpha=1.0, secondary_alphas=[0.5,1.5], generation_max_tokens=16,
                    generation_targets=['observed','er_1'], cases=cases, technical_exclusions=exclusions,
                    prepared_group_counts={k:len(v) for k,v in prepared.items()})
    write_frozen(OUT/'probe_manifest.json', manifest)
    return manifest


@torch.inference_mode()
def forward(model, ids, evidence, n, hidden=False):
    x = torch.tensor([ids], dtype=torch.long)
    valid = torch.ones_like(x)
    mask = build_evidence_hidden_attention_mask(valid, torch.tensor([evidence]), torch.float32) if hidden else valid
    logits = model(input_ids=x, attention_mask=mask, position_ids=torch.arange(len(ids))[None,:],
                   use_cache=False, num_logits_to_keep=n).logits[0, :, :model._probe_vocab_size]
    assert logits.shape[0] == n and torch.isfinite(logits).all()
    assert not torch.cuda.is_initialized()
    return logits


def entropy(z):
    lp = z.log_softmax(-1)
    return -(lp.exp()*lp).sum(-1)


def matched_logits(obs, er):
    # Label-free per-state matching is a stronger diagnostic control than a
    # globally calibrated temperature; it is not a deployable baseline recipe.
    target = entropy(er)
    lo = torch.full_like(target, math.log(0.01))
    hi = torch.full_like(target, math.log(100.0))
    assert torch.all(entropy(obs/lo.exp()[:,None]) <= target+1e-4)
    assert torch.all(entropy(obs/hi.exp()[:,None]) >= target-1e-4)
    for _ in range(30):
        mid = (lo+hi)/2
        h = entropy(obs/mid.exp()[:,None])
        lo = torch.where(h < target, mid, lo)
        hi = torch.where(h < target, hi, mid)
    tau = ((lo+hi)/2).exp()
    z = obs/tau[:,None]
    gap = (entropy(z)-target).abs()
    assert gap.max().item() < 1e-4, gap.max().item()
    return z, dict(temperatures=tau.tolist(), max_entropy_error=gap.max().item())


def stats(z, labels, content):
    lp = z.log_softmax(-1)
    vals = lp.gather(-1, torch.tensor(labels)[:,None]).squeeze(-1)
    body = vals[content]
    return dict(full_sum=vals.sum().item(), full_mean=vals.mean().item(),
                answer_sum=body.sum().item(), answer_mean=body.mean().item(),
                token_logps=vals.tolist(), entropy=entropy(z).tolist(),
                top_ids=z.argmax(-1).tolist())


def targets(obs, hid):
    er = obs+(obs-hid)
    matched, info = matched_logits(obs, er)
    return dict(observed=obs, hidden=hid, er_0_5=obs+0.5*(obs-hid), er_1=er,
                er_1_5=obs+1.5*(obs-hid), temperature_0_7=obs/0.7,
                entropy_matched=matched), info


def append_json(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())


def evaluate_case(model, tok, c, boundary, checks):
    pref = c if boundary == 'native' else c['early']
    scores, raw, encoded = {}, {}, {}
    answers = {'original': c['answer']}
    if c['group'] != 'correct_control': answers['gold'] = c['gold_candidate']
    for name, answer in answers.items():
        text = pref['leading']+answer+c['closing']
        e = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        labels = e['input_ids']
        content = [i for i,(a,b) in enumerate(e['offset_mapping']) if a < len(pref['leading']+answer) and b > len(pref['leading'])]
        ids = pref['prefix_ids']+labels[:-1]
        mask = pref['evidence_mask']+[0]*(len(labels)-1)
        t = time.time()
        obs = forward(model, ids, mask, len(labels))
        if boundary == 'native' and name == 'original':
            assert labels == c['original_labels']
            actual = obs.log_softmax(-1).gather(-1,torch.tensor(labels)[:,None]).squeeze(-1)
            errors = (actual-torch.tensor(c['saved_teacher_logps'])).abs()
            checks['native_logp_max_error'] = errors.max().item()
            checks['native_logp_mean_error'] = errors.mean().item()
            if errors.max().item() >= 0.5:
                return dict(excluded=True, reason='Native CPU FP32 versus saved FP16 max error >=0.5 nat', checks=checks)
        hid = forward(model, ids, mask, len(labels), hidden=True)
        if checks.get('run_null_mask', False):
            null = forward(model, ids, [0]*len(ids), len(labels), hidden=True)
            checks['null_mask_max_logit_error'] = (null-obs).abs().max().item()
            assert checks['null_mask_max_logit_error'] < 1e-3
            checks['run_null_mask'] = False
        zs, info = targets(obs,hid)
        scores[name] = {method: stats(z,labels,content) for method,z in zs.items()}
        scores[name]['entropy_match_details'] = info
        encoded[name] = dict(labels=labels, content=content)
        raw[name] = zs
        cache_path = OUT/'logits'/f"{c['review_id']}_{boundary}_{name}.pt"
        cache_path.parent.mkdir(exist_ok=True)
        torch.save(dict(observed=obs, hidden=hid, labels=labels, content=content,
                        manifest_sha256=digest(OUT/'probe_manifest.json')),cache_path)
        print('SCORED',c['review_id'],boundary,name,'seconds',round(time.time()-t,1),flush=True)
    if c['group'] == 'correct_control':
        scores['gold'] = scores['original']
    divergence = None
    if c['group'] != 'correct_control':
        gold_ids, wrong_ids = encoded['gold']['labels'], encoded['original']['labels']
        k = next(i for i,(g,w) in enumerate(zip(gold_ids,wrong_ids)) if g != w)
        assert gold_ids[:k] == wrong_ids[:k]
        shared_error = (raw['gold']['observed'][k]-raw['original']['observed'][k]).abs().max().item()
        hidden_error = (raw['gold']['hidden'][k]-raw['original']['hidden'][k]).abs().max().item()
        assert max(shared_error,hidden_error) < 1e-3, (shared_error, hidden_error)
        local = {name:(z[k,gold_ids[k]]-z[k,wrong_ids[k]]).item() for name,z in raw['gold'].items()}
        expected = local['observed']+(local['observed']-local['hidden'])
        assert abs(local['er_1']-expected) < 1e-4
        divergence = dict(position=k, gold_token=tok.decode([gold_ids[k]]), wrong_token=tok.decode([wrong_ids[k]]),
                          shared_prefix_max_error=shared_error, hidden_shared_prefix_max_error=hidden_error,
                          log_odds=local, caveat='Next-token binary diagnostic, not full-answer correctness')
    return dict(excluded=False, scores=scores, divergence=divergence, checks=checks)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--scores-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(8); torch.set_num_interop_threads(1); torch.manual_seed(20260914)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    manifest=prepare(tok)
    print('FROZEN',digest(OUT/'probe_manifest.json'),[(c['review_id'],c['group'],len(c['prefix_ids'])) for c in manifest['cases']],flush=True)
    if args.prepare_only:return
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    started=time.time()
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32,
        low_cpu_mem_usage=True,attn_implementation='eager').eval()
    model._probe_vocab_size = len(tok)
    assert len(tok) == 151665, 'Match restrict_to_tokenizer_vocab=true in source training'
    assert all(p.device.type=='cpu' and p.dtype==torch.float32 for p in model.parameters())
    print('LOADED_CPU_SECONDS',round(time.time()-started,1),flush=True)
    done={}
    score_path=OUT/'scores.jsonl'
    if score_path.exists():done={(r['key'],r['boundary']):r for l in score_path.read_text().splitlines() for r in [json.loads(l)]}
    for c in manifest['cases']:
        for boundary in ['native']+(['early'] if c['early_probe'] else []):
            if (c['key'],boundary) in done:continue
            if boundary=='early' and done[(c['key'],'native')]['excluded']:continue
            begin=time.time();print('START',c['review_id'],boundary,flush=True)
            result=evaluate_case(model,tok,c,boundary,dict(run_null_mask=not bool(done)))
            row=dict(key=c['key'],review_id=c['review_id'],group=c['group'],boundary=boundary,
                     seconds=time.time()-begin,**result)
            append_json(score_path,row);done[(c['key'],boundary)]=row
            print('DONE',c['review_id'],boundary,'excluded',row['excluded'],'seconds',round(row['seconds'],1),flush=True)
    if args.scores_only:return
    gen_path=OUT/'generation.jsonl'
    generated={(r['key'],r['target']) for l in gen_path.read_text().splitlines() for r in [json.loads(l)]} if gen_path.exists() else set()
    for c in manifest['cases']:
        if not c['free_generate'] or done[(c['key'],'native')]['excluded']:continue
        for target in manifest['generation_targets']:
            if (c['key'],target) in generated:continue
            new=[];begin=time.time();reason='max_tokens'
            for t in range(manifest['generation_max_tokens']):
                ids=c['prefix_ids']+new;mask=c['evidence_mask']+[0]*len(new)
                obs=forward(model,ids,mask,1)
                z=obs if target=='observed' else obs+(obs-forward(model,ids,mask,1,True))
                if t==0:
                    expected=done[(c['key'],'native')]['scores']['original'][target]['top_ids'][0]
                    assert z[0].argmax().item()==expected
                new.append(int(z[0].argmax()));text=tok.decode(new,skip_special_tokens=False)
                print('GEN_TOKEN',c['review_id'],target,t+1,repr(text),flush=True)
                if '</answer>' in text:reason='answer_closed';break
                if new[-1]==tok.eos_token_id:reason='eos';break
            text=tok.decode(new,skip_special_tokens=False)
            answer=text.split('</answer>')[0].strip() if reason=='answer_closed' else None
            row=dict(key=c['key'],review_id=c['review_id'],group=c['group'],target=target,
                ids=new,text=text,answer=answer,stop_reason=reason,
                reference_em=norm(answer) in [norm(g) for g in c['gold']] if answer is not None else False,
                seconds=time.time()-begin)
            append_json(gen_path,row);print('GENERATED',json.dumps(row,ensure_ascii=False),flush=True)
    (OUT/'COMPLETED.json').write_text(json.dumps(dict(completed=True,device='cpu',cuda_initialized=torch.cuda.is_initialized(),
        manifest_sha256=digest(OUT/'probe_manifest.json'),score_rows=len(done),
        generation_rows=len(gen_path.read_text().splitlines()) if gen_path.exists() else 0,
        script_sha256=digest(Path(__file__))),indent=2))
    print('COMPLETE_CPU',flush=True)


if __name__=='__main__':main()
