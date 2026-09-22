"""Freeze outcome-stratified, previously unreviewed cases before teacher probes."""
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
from audit_pure_opd_mainline import features
from sample_independent_history import old_indices

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/er_correction_cpu_20260914'
SEED = 'er-correction-cpu-20260914-v1'
TAG = '20260914-1220-coef1'


def sha(x):
    return hashlib.sha256(x.encode()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'sample_manifest.json').exists(), 'Never replace a frozen sample'
    frame = pd.read_parquet(ROOT / 'data/nq_hotpotqa_train_30k_no_cold_start/train.parquet')
    lookup = {(r['data_source'], int(r['extra_info']['index'])): r for r in frame.to_dict('records')}
    excluded = set(old_indices())
    previous = json.loads((ROOT / 'reports/independent_history_20260914/sample_manifest.json').read_text())
    excluded.update(c['index'] for c in previous['cases'])
    candidates, hashes, population = [], {}, []
    for method in ['opd', 'sod']:
        for step in range(41, 51):
            path = ROOT / f'verl_checkpoints/pure-{method}-05B-n1-{TAG}/opd_diagnostics/step_{step:06d}.jsonl.gz'
            rel = str(path.relative_to(ROOT))
            hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
            with gzip.open(path, 'rt') as f:
                header = json.loads(next(f))
                rows = [json.loads(l) for l in f]
            assert len(rows) == header['selected_sequences'] == 128
            for r in rows:
                source = lookup[(r['data_source'], int(r['index']))]
                gold = list(source['reward_model']['ground_truth']['target'])
                response = ''.join(r['token_texts'])
                feat = features(response, gold)
                assert feat['score'] == r['sequence_score']
                c = dict(key=f"{method}:{step}:{r['index']}", method=method, step=step,
                         index=int(r['index']), data_source=r['data_source'], source_file=rel,
                         question=source['question'], gold=gold, response=response, **feat)
                population.append({k: v for k, v in c.items() if k not in ['response', 'question', 'gold']})
                if c['index'] not in excluded:
                    candidates.append(c)
    # Global deduplication before outcome stratification prevents shared questions
    # from being treated as independent evidence across OPD/SOD samples.
    unique = {}
    for c in sorted(candidates, key=lambda c: sha(SEED + '|dedup|' + c['key'])):
        unique.setdefault(c['index'], c)
    selected, counts = [], {}
    for method in ['opd', 'sod']:
        for score, count in [(0, 16), (1, 4)]:
            pool = [c for c in unique.values() if c['method'] == method and c['score'] == score]
            pool.sort(key=lambda c: sha(SEED + '|sample|' + c['key']))
            counts[f'{method}:{score}'] = len(pool)
            assert len(pool) >= count
            selected.extend(pool[:count])
    selected.sort(key=lambda c: sha(SEED + '|blind|' + c['key']))
    cards = []
    for i, c in enumerate(selected, 1):
        c['review_id'] = f'E{i:03d}'
        cards.append({k: c[k] for k in ['review_id', 'question', 'gold', 'response']})
    manifest = dict(seed=SEED, window=[41, 50], excluded_prior_indices=sorted(excluded),
                    source_hashes=hashes, pool_counts=counts, cases=selected)
    (OUT / 'sample_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (OUT / 'review_cards.json').write_text(json.dumps(cards, ensure_ascii=False, indent=2))
    (OUT / 'population.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in population))
    (OUT / 'REVIEW_CARDS.md').write_text('\n\n'.join(
        f"## {c['review_id']}\n\n{c['question']}\n\nReference: {c['gold']}\n\n{c['response']}" for c in cards))
    print(json.dumps(dict(n_population=len(population), n_sample=len(selected), pool_counts=counts,
                         manifest_sha256=hashlib.sha256((OUT / 'sample_manifest.json').read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
