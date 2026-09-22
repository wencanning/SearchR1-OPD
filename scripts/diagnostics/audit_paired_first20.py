"""CPU-only descriptive audit; lexical hits are NOT evidence sufficiency labels."""
import gzip
import hashlib
import json
import re
import string
from collections import Counter
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/paired_coef1_first20'
TAG = '20260914-1220-coef1'


def norm(s):
    s = ''.join(c for c in s.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split())


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(ROOT / 'data/nq_hotpotqa_train_30k_no_cold_start/train.parquet')
    lookup = {}
    for r in df.to_dict('records'):
        key = (r['data_source'], int(r['extra_info']['index']))
        assert key not in lookup
        lookup[key] = r
    summary, per_step, all_rows, reviewed = {}, [], [], []
    for method in ['opd', 'sod']:
        rows = []
        for step in range(1, 21):
            path = ROOT / f'verl_checkpoints/pure-{method}-05B-n1-{TAG}/opd_diagnostics/step_{step:06d}.jsonl.gz'
            with gzip.open(path, 'rt') as f:
                header = json.loads(next(f))
                records = [json.loads(line) for line in f]
            assert len(records) == header['batch_size'] == header['selected_sequences'] == 128
            for r in records:
                source = lookup[(r['data_source'], int(r['index']))]
                targets = list(source['reward_model']['ground_truth']['target'])
                assert targets == r['ground_truth_targets']
                text = r['decoded_response']
                infos = re.findall(r'<information>(.*?)</information>', text, re.S)
                targets_norm = [norm(t) for t in targets if norm(t)]
                hits = [b for b in infos if any(re.search(r'(?<!\w)' + re.escape(t) + r'(?!\w)', norm(b)) for t in targets_norm)]
                prompt = ''.join(p['content'] for p in source['prompt'])
                answers = re.findall(r'<answer>(.*?)</answer>', prompt + text, re.S)
                score = int(len(answers) > 1 and norm(answers[-1]) in targets_norm)
                row = {k:r[k] for k in ['global_step','sequence_index','index','data_source',
                       'final_answer','answer_correct','sequence_score','retrieval_hit']}
                row.update(method=method, question=source['question'], gold=targets,
                           recomputed_score=score, bounded_gold_hit=bool(hits),
                           n_search=len(infos), missing_answer=r['final_answer'] is None,
                           question_id=source['id'], decoded_response=text, hit_blocks=hits,
                           source_file=str(path.relative_to(ROOT)))
                rows.append(row)
        def stats(rs):
            failed = [r for r in rs if r['sequence_score'] == 0]
            hit = [r for r in rs if r['bounded_gold_hit']]
            return dict(n=len(rs), correct=sum(r['sequence_score']==1 for r in rs),
                failed=len(failed), missing_answer=sum(r['missing_answer'] for r in rs),
                raw_hit=sum(bool(r['retrieval_hit']) for r in rs), bounded_hit=len(hit),
                bounded_hit_failed=sum(r['sequence_score']==0 for r in hit),
                bounded_hit_missing=sum(r['missing_answer'] for r in hit),
                reward_recompute_mismatch=sum(r['sequence_score']!=r['recomputed_score'] for r in rs),
                diagnostic_vs_reward_mismatch=sum(r['answer_correct'] is not None and
                    bool(r['answer_correct'])!=bool(r['sequence_score']) for r in rs),
                unique_questions=len({r['question_id'] for r in rs}))
        summary[method] = stats(rows)
        for step in range(1,21):
            per_step.append(dict(method=method, step=step, **stats([r for r in rows if r['global_step']==step])))
        # Fixed reproducible sample, not hand-picked by how compelling it looks.
        for low,high in [(1,10),(11,20)]:
            candidates=[r for r in rows if low<=r['global_step']<=high and
                        r['bounded_gold_hit'] and r['sequence_score']==0 and not r['missing_answer']]
            candidates.sort(key=lambda r:hashlib.sha256(
                f"audit42|{method}|{r['global_step']}|{r['question_id']}".encode()).hexdigest())
            reviewed.extend(candidates[:3])
        all_rows.extend(rows)
    a={(r['global_step'],r['index']):r for r in all_rows if r['method']=='opd'}
    b={(r['global_step'],r['index']):r for r in all_rows if r['method']=='sod'}
    assert len(a)==len(b)==2560 and a.keys()==b.keys()
    summary['paired'] = dict(n=len(a), opd_only_correct=sum(a[k]['sequence_score']>b[k]['sequence_score'] for k in a),
                            sod_only_correct=sum(b[k]['sequence_score']>a[k]['sequence_score'] for k in a))
    for i,r in enumerate(reviewed,1):
        r['review_id']=i
        other=(b if r['method']=='opd' else a)[(r['global_step'],r['index'])]
        r['other_method_answer']=other['final_answer']
        r['other_method_score']=other['sequence_score']
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    pd.DataFrame(per_step).to_csv(OUT/'per_step.csv',index=False)
    with (OUT/'trajectories.jsonl').open('w') as f:
        for r in all_rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    (OUT/'review_sample.json').write_text(json.dumps(reviewed,ensure_ascii=False,indent=2))
    print(json.dumps(summary,indent=2))
    print('Manual sample: 3 per method per half (12 total), SHA256 audit42 order.')


if __name__=='__main__':
    run()
