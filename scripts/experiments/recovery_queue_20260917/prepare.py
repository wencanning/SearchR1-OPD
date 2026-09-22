"""Freeze a fresh train-only collection pool; never edit dataset references."""
import csv
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/recovery_queue_20260917'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm(x):
    return ' '.join(x.lower().split())


def main():
    import pyarrow.parquet as pq
    sources = [
        'reports/er_next_20260916/fork_states.frozen.json',
        'reports/evidence_use_gpu5_20260915/panel.json',
        'reports/evidence_stopping_gpu5_20260915/candidates.json',
        'reports/evidence_gap_gpu23_20260915/panels.json',
        'reports/recovery_20260917/train_inputs.frozen.json',
        'reports/recovery_20260917/probe_inputs.frozen.json',
        'reports/recovery_20260917_native/train_inputs.frozen.json',
    ]
    excluded = set()
    def visit(x):
        if isinstance(x, dict):
            if isinstance(x.get('question'), str):
                excluded.add(norm(x['question']))
            for v in x.values(): visit(v)
        elif isinstance(x, list):
            for v in x: visit(v)
    hashes = {}
    for name in sources:
        p = ROOT / name
        if p.exists():
            visit(json.loads(p.read_text())); hashes[name] = digest(p)
    trainfile = ROOT / 'data/nq_hotpotqa_train_30k_no_cold_start/train.parquet'
    # Exclude every known test question, not merely previously measured cases.
    for name in ['data/nq_hotpotqa_train/test.parquet',
                 'data/nq_hotpotqa_train_30k_no_cold_start/validation_diagnostic_512.parquet']:
        p = ROOT / name
        excluded.update(norm(r['question']) for r in pq.read_table(p, columns=['question']).to_pylist())
        hashes[name] = digest(p)
    hashes[str(trainfile.relative_to(ROOT))] = digest(trainfile)
    rows = pq.read_table(trainfile).to_pylist()
    eligible = {}
    rejected = []
    for r in rows:
        q = norm(r['question'])
        if r['data_source'] != 'hotpotqa' or q in excluded: continue
        gold = list(r['golden_answers'])
        if not gold or any(not isinstance(a, str) or not a.strip() for a in gold):
            rejected.append({'id': str(r['id']), 'reason': 'empty_reference'}); continue
        # Input-only diagnostic cohort restriction, explicitly not factual QA.
        if any(len(a.split()) > 20 for a in gold):
            rejected.append({'id': str(r['id']), 'reason': 'reference_over_20_words'}); continue
        assert any(r['question'] in p['content'] for p in r['prompt'])
        eligible[q] = r
    ordered = sorted(eligible, key=lambda q: hashlib.sha256(('recovery_queue_20260917|' + q).encode()).hexdigest())
    assert len(ordered) >= 128
    cases = [dict(qid=str(eligible[q]['id']), split='train', question=eligible[q]['question'],
                  gold=list(eligible[q]['golden_answers']), prompt=[dict(p) for p in eligible[q]['prompt']])
             for q in ordered[:128]]
    assert len({c['qid'] for c in cases}) == len(cases)
    assert len({norm(c['question']) for c in cases}) == len(cases)
    assert not any(norm(c['question']) in excluded for c in cases)
    payload = dict(cases=cases, replicates=4, max_segment=256, max_generated=768,
                   max_new_searches=3, max_context=4096, topk=3, observation_tokens=512,
                   seed=2026091747, prepare_sha256=digest(Path(__file__)), source_sha256=hashes,
                   qualification='Fresh 128-question HotpotQA TRAIN-only diagnostic collection; four samples per model at eligible first-retrieval states. No new heldout evaluation and no training. Actual native parser/retry under bounded diagnostic caps, not full production rollout.',
                   reference_audit='Source labels preserved. Nonempty/short-reference and prompt checks only; factual correctness and semantic aliases NOT fully certified. Review before training; never claim this pool is a clean benchmark.',
                   rejected_input_only=rejected)
    OUT.mkdir(exist_ok=True)
    p = OUT / 'train_inputs.frozen.json'
    if p.exists(): assert json.loads(p.read_text()) == payload
    else: p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    with (OUT / 'input_review.tsv').open('w') as f:
        w = csv.writer(f, delimiter='\t'); w.writerow(['qid', 'question', 'source_answers'])
        w.writerows((c['qid'], c['question'], json.dumps(c['gold'], ensure_ascii=False)) for c in cases)
    print(json.dumps({'train_questions':len(cases), 'eval_questions':0, 'replicates':4,
                      'excluded_questions':len(excluded), 'input_sha256':digest(p)}))


if __name__ == '__main__': main()
