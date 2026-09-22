"""Small frozen-teacher history intervention; CPU only, no training/retrieval."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
import argparse
import gzip
import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/history_intervention_cpu_20260914'
MODEL = '/data/home/wencanning/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3'
TAG = '20260914-1220-coef1'
CASES = [
    # Chosen before any intervention scores; no substitutions after exclusion.
    ('opd', 70, 58559, 'failure_missing_evidence', 'Apostolo Zeno and Pietro Pariati', 'Remo Giazotto'),
    ('opd', 63, 23560, 'failure_sufficient_evidence', 'The Used', 'Flight of the Conchords'),
    ('sod', 67, 82035, 'failure_sufficient_evidence', 'the founding of the international Scouting Movement', 'Scoutcraft'),
    ('sod', 70, 78463, 'correct_control', '86', '2'),
    ('opd', 63, 43500, 'correct_control', 'First World War', 'Imperial War'),
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare(tok):
    frame = pd.read_parquet(ROOT / 'data/nq_hotpotqa_train_30k_no_cold_start/train.parquet')
    lookup = {(r['data_source'], int(r['extra_info']['index'])): r for r in frame.to_dict('records')}
    output = []
    for method, step, index, group, gold, wrong in CASES:
        path = ROOT / f'verl_checkpoints/pure-{method}-05B-n1-{TAG}/opd_diagnostics/step_{step:06d}.jsonl.gz'
        with gzip.open(path, 'rt') as f:
            next(f)
            r = next(r for line in f for r in [json.loads(line)] if int(r['index']) == index)
        source = lookup[('hotpotqa', index)]
        assert gold in list(source['reward_model']['ground_truth']['target'])
        observed = gold if group == 'correct_control' else wrong
        assert r['final_answer'].strip() == observed
        joined = ''.join(r['token_texts'])
        match = list(re.finditer(r'<answer>(.*?)</answer>', joined, re.S))[-1]
        start = match.start(1)
        while joined[start].isspace():
            start += 1
        bounds, pos = [], 0
        for token in r['token_texts']:
            bounds.append((pos, pos + len(token)))
            pos += len(token)
        cut = next(i for i, (a, b) in enumerate(bounds) if a <= start < b)
        lead = joined[bounds[cut][0]:start]
        assert not any(c in lead for c in '<>'), 'Answer boundary overlaps protocol tag'
        text = ''.join(r['token_texts'][:cut])
        spans = list(re.finditer(r'<think>(.*?)</think>', text, re.S))
        assert len(spans) >= 2
        # The protocol opens think after tool feedback. Observation text is never edited.
        post, pre = [], []
        for n, span in enumerate(spans):
            body_a, body_b = span.span(1)
            ids = [i for i, (a, b) in enumerate(bounds[:cut])
                   if a >= body_a and b <= body_b and r['loss_mask'][i]]
            (pre if n == 0 else post).extend(ids)
        assert post and pre and set(post).isdisjoint(pre)
        assert all(r['loss_mask'][i] for i in post + pre)
        prompt_ids = tok.apply_chat_template(list(source['prompt']), add_generation_prompt=True, tokenize=True)
        response_ids = r['token_ids'][:cut]
        assert tok(lead + observed + '\n</answer>', add_special_tokens=False)['input_ids'][0] == r['token_ids'][cut]
        original_ids = prompt_ids + response_ids
        pre_mask, post_mask = [1] * len(original_ids), [1] * len(original_ids)
        for i in pre:
            pre_mask[len(prompt_ids) + i] = 0
        for i in post:
            post_mask[len(prompt_ids) + i] = 0
        # Token deletion is a secondary sensitivity test, not the position-controlled primary.
        removed = set(post)
        deleted_ids = prompt_ids + [t for i, t in enumerate(response_ids) if i not in removed]
        conditions = {
            'original': dict(ids=original_ids, mask=[1]*len(original_ids)),
            'hide_post_think': dict(ids=original_ids, mask=post_mask),
            'hide_pre_think_control': dict(ids=original_ids, mask=pre_mask),
            'delete_post_think': dict(ids=deleted_ids, mask=[1]*len(deleted_ids)),
        }
        assert conditions['original']['ids'] == conditions['hide_post_think']['ids'] == conditions['hide_pre_think_control']['ids']
        output.append(dict(key=f'{method}:{step}:{index}', method=method, step=step, index=index, group=group,
            question=source['question'], gold=gold, wrong=wrong, observed=observed, leading=lead,
            source_file=str(path.relative_to(ROOT)), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            prefix_text=text, post_think_texts=[s.group(1) for s in spans[1:]], pre_think_text=spans[0].group(1),
            post_hidden_tokens=len(post), pre_hidden_tokens=len(pre),
            malformed_information=text.count('<information>') != text.count('</information>'),
            baseline_saved_logp=r['teacher_log_prob'][cut], baseline_token=r['token_texts'][cut],
            conditions=conditions))
    return output


@torch.inference_mode()
def evaluate(model, tok, condition, leading, answer):
    encoded = tok(leading + answer + '\n</answer>', add_special_tokens=False, return_offsets_mapping=True)
    labels = encoded['input_ids']
    content_end = len(leading + answer)
    content_indices = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if a < content_end and b > len(leading)]
    x = torch.tensor([condition['ids'] + labels[:-1]], dtype=torch.long)
    mask = torch.tensor([condition['mask'] + [1] * (len(labels) - 1)], dtype=torch.long)
    logits = model(input_ids=x, attention_mask=mask, position_ids=torch.arange(x.shape[1])[None, :],
                   use_cache=False, num_logits_to_keep=len(labels)).logits[0]
    assert torch.isfinite(logits).all()
    logp = logits.log_softmax(-1)
    selected = logp.gather(-1, torch.tensor(labels)[:, None]).squeeze(-1)
    vals = selected[content_indices]
    top = int(logp[0].argmax())
    return dict(full_sum=selected.sum().item(), full_mean=selected.mean().item(),
                answer_sum=vals.sum().item(), answer_mean=vals.mean().item(),
                first_logp=selected[0].item(), first_top_token=tok.decode([top]),
                first_top_logp=logp[0, top].item(), token_ids=labels,
                token_logps=selected.tolist(), n_content=len(content_indices), n_full=len(labels))


def summarize():
    rows = [json.loads(l) for l in (OUT / 'scores.jsonl').read_text().splitlines()]
    summary = []
    for row in rows:
        margins = {condition: {metric: vals['gold'][metric] - vals['wrong'][metric]
                              for metric in ['full_sum', 'full_mean', 'answer_sum', 'answer_mean']}
                   for condition, vals in row['scores'].items()}
        summary.append(dict(key=row['key'], group=row['group'], gold=row['gold'], wrong=row['wrong'],
                            baseline_error=row['baseline_error'], margins=margins))
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize()
        return
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    cases = prepare(tok)
    manifest = dict(model=MODEL, torch=torch.__version__, transformers=transformers.__version__,
                    dtype='float32', attention='eager', device='cpu', cache=False, threads=8, cases=cases)
    mpath = OUT / 'manifest.json'
    if mpath.exists():
        assert digest(json.loads(mpath.read_text())) == digest(manifest), 'Existing manifest differs; use a new run directory'
    else:
        mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('MANIFEST', digest(manifest), [(c['key'], c['post_hidden_tokens'], c['pre_hidden_tokens']) for c in cases], flush=True)
    if args.prepare_only:
        return
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32,
                low_cpu_mem_usage=True, attn_implementation='eager').eval()
    assert next(model.parameters()).device.type == 'cpu'
    print('CPU MODEL LOADED', flush=True)
    done = set()
    for filename in ['scores.jsonl', 'excluded.jsonl']:
        path = OUT / filename
        if path.exists():
            done.update(json.loads(l)['key'] for l in path.read_text().splitlines())
    for case in cases:
        if case['key'] in done:
            continue
        begin = time.time()
        result = {k: case[k] for k in ['key', 'group', 'question', 'gold', 'wrong', 'post_hidden_tokens', 'pre_hidden_tokens']}
        result['scores'] = {}
        original_name = 'gold' if case['group'] == 'correct_control' else 'wrong'
        print('START', case['key'], flush=True)
        baseline = evaluate(model, tok, case['conditions']['original'], case['leading'], case[original_name])
        error = abs(baseline['first_logp'] - case['baseline_saved_logp'])
        result['baseline_error'] = error
        result['baseline_saved_logp'] = case['baseline_saved_logp']
        print('BASELINE', case['key'], error, flush=True)
        if error >= .5:
            with (OUT / 'excluded.jsonl').open('a') as f:
                f.write(json.dumps(dict(key=case['key'], reason='CPU/saved first-token mismatch >= .5 nat', error=error)) + '\n')
            continue
        for cname, condition in case['conditions'].items():
            scores = {}
            for candidate in ['gold', 'wrong']:
                scores[candidate] = baseline if cname == 'original' and candidate == original_name else evaluate(
                    model, tok, condition, case['leading'], case[candidate])
                print('SCORED', case['key'], cname, candidate, 'elapsed', round(time.time()-begin, 1), flush=True)
            result['scores'][cname] = scores
            print('MARGIN', cname, scores['gold']['full_sum'] - scores['wrong']['full_sum'], flush=True)
        result['seconds'] = time.time()-begin
        with (OUT / 'scores.jsonl').open('a') as f:
            f.write(json.dumps(result, ensure_ascii=False)+'\n')
        print('DONE', case['key'], round(result['seconds'], 1), flush=True)
    summarize()


if __name__ == '__main__':
    main()
