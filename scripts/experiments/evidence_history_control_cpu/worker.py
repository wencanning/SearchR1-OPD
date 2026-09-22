"""Frozen-history teacher diagnostic; CPU only, no retrieval or training."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/evidence_history_control_cpu_20260915'
spec = importlib.util.spec_from_file_location('previous_cpu', ROOT / 'scripts/experiments/evidence_cpu_followup/worker.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
M = P.M
CONDITIONS = ['first_hop', 'both_hops', 'with_distractors']
NEUTRAL = '<think>I will use the available information to decide the next step.</think>\n'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(tok):
    previous = M.read(P.OUT / 'inputs.frozen.json')
    source = P.SOURCE
    protocol = M.read(source / 'protocol.frozen.json')
    eligible = set(protocol['selected_qids']) - set(previous['selected_teacher_qids'])
    chosen = sorted(eligible, key=lambda q: hashlib.sha256(('fixed-history-20260915|' + q).encode()).hexdigest())[:8]
    assert len(chosen) == 8
    candidates = {q['qid']: q for q in M.read(source / 'candidates.json')}
    original = {(r['label'], r['qid'], r['condition']): r for r in P.rows(source / 'first_actions.jsonl')}
    cases = []
    for qid in chosen:
        record = original['opd-step100', qid, 'first_hop']
        boundary = P.make_boundary(tok, record, '')
        assert boundary is not None, ('Missing boundary; do not replace sample', qid)
        actual_ids = boundary['ids'][len(record['input_ids']):]
        for history, history_ids in [('neutral', tok.encode(NEUTRAL, add_special_tokens=False)), ('fixed_student', actual_ids)]:
            for condition in CONDITIONS:
                prefix, _ = M.make_prefix(tok, candidates[qid], condition)
                ids = prefix + history_ids
                text = tok.decode(prefix, skip_special_tokens=False)
                start, end = list(re.finditer(r'<information>(.*?)</information>', text, re.S))[-1].span(1)
                lengths = [0] + [len(tok.decode(prefix[:i], skip_special_tokens=False)) for i in range(1, len(prefix) + 1)]
                mask = [int(b > a and a >= start and b <= end) for a, b in zip(lengths, lengths[1:])] + [0] * len(history_ids)
                assert len(ids) == len(mask) and sum(mask) > 0 and len(ids) + 8 < 4096
                cases.append(dict(qid=qid, history=history, condition=condition, ids=ids, evidence_mask=mask,
                                  history_ids=history_ids, history_text=tok.decode(history_ids, skip_special_tokens=False)))
    tags = {s: tok.encode('<' + s + '>', add_special_tokens=False) for s in ['answer', 'search']}
    assert all(tok.decode(ids) == '<' + key + '>' for key, ids in tags.items())
    assert tags['answer'][0] == tags['search'][0] and len(tags['answer']) == len(tags['search']) == 3
    for qid in chosen:
        for h in ['neutral', 'fixed_student']:
            assert len({tuple(c['history_ids']) for c in cases if c['qid'] == qid and c['history'] == h}) == 1
    sources = [source / 'protocol.frozen.json', source / 'first_actions.jsonl', source / 'candidates.json',
               source / 'review.json', P.OUT / 'inputs.frozen.json', Path(P.__file__), Path(M.__file__),
               ROOT / 'verl/utils/evidence_residual.py', Path(__file__)]
    manifest = dict(cases=cases, selected_qids=chosen, excluded_teacher_qids=previous['selected_teacher_qids'],
                    selection='First 8 by SHA256(fixed-history-20260915|qid), excluding prior 6 teacher qids; no outcome filtering',
                    source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources}, tags=tags,
                    alphas=[0, .5, 1, 1.5], temperatures=[.5, 1, 2], primary_alpha=1.5,
                    device='cpu', dtype='float32', threads=4, environment='evidence-stopping-cpu@a33f1b1d',
                    limitation='Canonical opening-tag path probabilities; not all action realizations or answer accuracy. Previously studied student questions, new teacher questions.')
    path = OUT / 'inputs.frozen.json'
    if path.exists():
        assert M.read(path) == manifest, 'Frozen input/code mismatch'
    else:
        M.save(path, manifest)
    return manifest


def run(manifest, torch):
    from transformers import AutoModelForCausalLM
    from verl.utils.evidence_residual import build_evidence_hidden_attention_mask
    path = OUT / 'teacher_paths.jsonl'
    existing = P.rows(path)
    done = {(r['qid'], r['history'], r['condition'], r['action']) for r in existing}
    assert len(done) == len(existing)
    model = AutoModelForCausalLM.from_pretrained(P.TEACHER, local_files_only=True, torch_dtype=torch.float32,
                                                attn_implementation='eager', low_cpu_mem_usage=True).eval().to('cpu')
    assert all(p.device.type == 'cpu' for p in model.parameters())
    print('LOAD_CPU_TEACHER_COMPLETE', flush=True)

    def forward(ids, mask, keep):
        x = torch.tensor([ids], device='cpu')
        valid = torch.ones_like(x)
        attn = valid if mask is None else build_evidence_hidden_attention_mask(valid, torch.tensor([mask]), torch.float32)
        with torch.inference_mode():
            z = model(x, attention_mask=attn, position_ids=torch.arange(len(ids))[None, :],
                      use_cache=False, num_logits_to_keep=keep).logits[0].float()
        assert z.shape[0] == keep and torch.isfinite(z).all() and not torch.cuda.is_initialized()
        return z

    def score(z, tag):
        lp = z.log_softmax(-1)
        values = lp[torch.arange(len(tag)), torch.tensor(tag)]
        return dict(log_probability=float(values.sum()), token_log_probabilities=values.tolist(),
                    token_entropies=(-(lp.exp() * lp).sum(-1)).tolist())

    for c in manifest['cases']:
        for action, tag in manifest['tags'].items():
            key = (c['qid'], c['history'], c['condition'], action)
            if key in done:
                continue
            tic = time.time()
            # Last len(tag) rows predict the first tag token through its final token.
            ids = c['ids'] + tag[:-1]
            mask = c['evidence_mask'] + [0] * (len(tag) - 1)
            obs = forward(ids, None, len(tag))
            hid = forward(ids, mask, len(tag))
            if not (OUT / 'NULL_MASK_CHECK.json').exists():
                zero = forward(ids, [0] * len(ids), len(tag))
                error = float((obs - zero).abs().max())
                assert error < 1e-4, error
                # A separate prefix-only forward verifies the causal row alignment.
                prefix_only = forward(c['ids'], None, 1)
                alignment_error = float((obs[0] - prefix_only[0]).abs().max())
                assert alignment_error < 1e-3, alignment_error
                M.save(OUT / 'NULL_MASK_CHECK.json', dict(passed=True, max_error=error,
                       causal_alignment_max_error=alignment_error, cuda_initialized=False))
            result = dict(qid=c['qid'], history=c['history'], condition=c['condition'], action=action,
                          targets={str(a): score(obs + a * (obs - hid), tag) for a in manifest['alphas']},
                          hidden=score(hid, tag), temperatures={str(t): score(obs / t, tag) for t in manifest['temperatures']},
                          prefix_tokens=len(c['ids']), evidence_tokens=sum(mask), seconds=time.time() - tic,
                          device='cpu', dtype='float32', cuda_initialized=False)
            P.append(path, result)
            done.add(key)
            M.save(OUT / 'status.json', dict(state='running', completed_paths=len(done), total_paths=96, cuda_initialized=False))
            print('CPU_PATH', len(done), 96, *key, round(result['seconds'], 2), flush=True)
    assert len(P.rows(path)) == 96
    M.save(OUT / 'status.json', dict(state='inference_complete', completed_paths=96, total_paths=96, cuda_initialized=False))


def main():
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260915)
    assert not torch.cuda.is_initialized()
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run'])
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(ROOT / 'data/student/0.5B', local_files_only=True)
    manifest = prepare(tok)
    print('PREPARED', len(manifest['cases']), 'states', manifest['selected_qids'], flush=True)
    if args.stage == 'run':
        run(manifest, torch)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        M.save(OUT / 'failure.json', dict(error=repr(exc), at=time.strftime('%Y-%m-%d %H:%M:%S')))
        raise
