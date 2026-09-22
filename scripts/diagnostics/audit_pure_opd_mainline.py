"""CPU-only exploratory audit. Lexical matches are not evidence-sufficiency labels.

Run from any directory; freezes OPD/SOD at step 70 and DGPO at local step 165.
Does not import torch, contact a retriever, or change training configurations.
"""
import gzip
import hashlib
import json
import re
import statistics
import string
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/pure_opd_mainline_20260914"
TAG = "20260914-1220-coef1"
DGPO = ROOT / "wandb/run-20260903_102107-ykcj0ax6/files/output.log"


def norm(s):
    s = ''.join(c for c in s.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split())


def hit(text, gold):
    text = norm(text)
    return any(re.search(r'(?<!\w)' + re.escape(norm(g)) + r'(?!\w)', text)
               for g in gold if norm(g))


def features(text, gold):
    queries = re.findall(r'<search>(.*?)</search>', text, re.S)
    infos = re.findall(r'<information>(.*?)</information>', text, re.S)
    thoughts = re.findall(r'<think>(.*?)</think>', text, re.S)
    answers = re.findall(r'<answer>(.*?)</answer>', text, re.S)
    answer = answers[-1].strip() if answers else None
    qnorm = [norm(q) for q in queries]
    score = int(answer is not None and norm(answer) in [norm(g) for g in gold])
    return dict(answer=answer, score=score, searches=len(queries),
                repeated_query=len(qnorm) > len(set(qnorm)),
                bounded_gold_hit=any(hit(i, gold) for i in infos),
                thought_gold_hit=any(hit(t, gold) for t in thoughts),
                missing_answer=answer is None,
                malformed_information=text.count('<information>') != text.count('</information>'))


def stats(rows):
    n = len(rows)
    hits = [r for r in rows if r['bounded_gold_hit']]
    return dict(n=n, correct=sum(r['score'] for r in rows),
                missing=sum(r['missing_answer'] for r in rows),
                searches_mean=statistics.mean(r['searches'] for r in rows),
                single_search=sum(r['searches'] == 1 for r in rows),
                repeated_query=sum(r['repeated_query'] for r in rows),
                bounded_hit=len(hits), bounded_hit_wrong=sum(not r['score'] for r in hits),
                thought_hit_wrong=sum(r['thought_gold_hit'] and not r['score'] for r in rows),
                malformed_information=sum(r['malformed_information'] for r in rows),
                by_search={str(k):dict(n=len(rs), correct=sum(r['score'] for r in rs))
                           for k in sorted({r['searches'] for r in rows})
                           for rs in [[r for r in rows if r['searches'] == k]]})


def sample(rows, key, count):
    return sorted(rows, key=lambda r: hashlib.sha256(
        f"mainline42|{key}|{r['method']}|{r['step']}|{r['index']}".encode()).hexdigest())[:count]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(ROOT / 'data/nq_hotpotqa_train_30k_no_cold_start/train.parquet')
    lookup = {(r['data_source'], int(r['extra_info']['index'])): r
              for r in train.to_dict('records')}
    qlookup = {r['question'].strip(): r for r in train.to_dict('records')}
    val = pd.read_parquet(ROOT / 'data/nq_hotpotqa_train_30k_no_cold_start/validation_diagnostic_512.parquet')
    counts = val['data_source'].value_counts().to_dict()
    assert sum(counts.values()) == 512
    metrics = []
    for method in ['opd', 'sod', 'dgpo']:
        path = DGPO if method == 'dgpo' else ROOT / f'verl_checkpoints/pure-{method}-05B-n1-{TAG}/train.log'
        for line in path.open():
            match = re.search(r'step:(\d+) - ', line)
            if not match or int(match[1]) > (165 if method == 'dgpo' else 70):
                continue
            row = {k: float(v) for k, v in re.findall(r'([\w/]+):(-?[\d.]+)', line)}
            row.update(method=method, step=int(match[1]))
            if 'val/test_score/Avg' in row:
                # Recover integer counts from console values rounded to .001.
                cs = {k: round(n * row['val/test_score/' + k]) for k, n in counts.items()}
                assert all(abs(cs[k] / n - row['val/test_score/' + k]) <= .000501
                           for k, n in counts.items())
                row['val/micro_correct'] = sum(cs.values())
                row['val/micro_em'] = sum(cs.values()) / 512
            metrics.append(row)
    pd.DataFrame(metrics).to_csv(OUT / 'metrics.csv', index=False)
    summary = {'validation_counts': counts, 'curves': [], 'trajectories': [],
               'dgpo_log_coverage': {}, 'paired': {}}
    keys = ['critic/score/mean', 'state_tokens/total', 'response_length/mean',
            'env/number_of_valid_search', 'env/finish_ratio', 'opd/student_entropy',
            'actor/entropy_loss', 'opd/divergence', 'opd/reverse_kl_k1']
    for method in ['opd', 'sod', 'dgpo']:
        for lo, hi in [(1, 10), (31, 40), (61, 70), (151, 160)]:
            rs = [r for r in metrics if r['method'] == method and lo <= r['step'] <= hi]
            if rs:
                summary['curves'].append(dict(method=method, window=[lo, hi], n_steps=len(rs),
                    **{k: statistics.mean(r[k] for r in rs if k in r) for k in keys if k in rs[0]}))
    review, pair = [], {}
    with (OUT / 'trajectory_features.jsonl').open('w') as out:
        for method in ['opd', 'sod']:
            for lo, hi in [(1, 10), (31, 40), (61, 70)]:
                rows = []
                for step in range(lo, hi + 1):
                    path = ROOT / f'verl_checkpoints/pure-{method}-05B-n1-{TAG}/opd_diagnostics/step_{step:06d}.jsonl.gz'
                    with gzip.open(path, 'rt') as f:
                        header = json.loads(next(f))
                        assert header['batch_size'] == header['selected_sequences'] == 128
                        n = 0
                        for line in f:
                            n += 1
                            r = json.loads(line)
                            source = lookup[(r['data_source'], int(r['index']))]
                            gold = list(source['reward_model']['ground_truth']['target'])
                            text = r['decoded_response']
                            row = dict(method=method, step=step, index=r['index'], question=source['question'],
                                       gold=gold, source_file=str(path.relative_to(ROOT)), **features(text, gold))
                            assert row['score'] == r['sequence_score']
                            # Keep descriptive token metrics separate from semantic labels.
                            for seg in ['think', 'search', 'answer', 'tag']:
                                ids = [i for i, (s, m) in enumerate(zip(r['segments'], r['loss_mask'])) if s == seg and m]
                                row[seg + '_tokens'] = len(ids)
                                if ids:
                                    row[seg + '_mean_gap'] = statistics.mean(r['logp_gap_teacher_minus_student'][i] for i in ids)
                                    row[seg + '_mean_entropy'] = statistics.mean(r['student_entropy'][i] for i in ids)
                            row['policy_tokens'] = sum(r['loss_mask'])
                            out.write(json.dumps(row, ensure_ascii=False) + '\n')
                            pair.setdefault((step, int(r['index'])), {})[method] = row['score']
                            row['decoded_response'] = text
                            rows.append(row)
                        assert n == 128
                summary['trajectories'].append(dict(method=method, window=[lo, hi], **stats(rows)))
                if lo == 61:
                    for group, candidates in [
                        ('all_wrong', [r for r in rows if not r['score'] and not r['missing_answer']]),
                        ('hit_wrong', [r for r in rows if not r['score'] and r['bounded_gold_hit'] and not r['missing_answer']])]:
                        for row in sample(candidates, group, 6):
                            review.append(dict(review_group=group, **row))
                print(method, lo, hi, 'done', flush=True)
    assert all(set(v) == {'opd', 'sod'} for v in pair.values())
    summary['paired'] = dict(n=len(pair), opd_only=sum(v['opd'] > v['sod'] for v in pair.values()),
                            sod_only=sum(v['sod'] > v['opd'] for v in pair.values()))
    # DGPO diagnostics were disabled. Recover only randomly printed training examples;
    # validation steps 50/100/150 are excluded to prevent train/validation mixing.
    printed = []
    text = DGPO.read_text()
    for match in re.finditer(r'epoch \d+, step (\d+)\n(.*?)(?=\nepoch \d+, step |\Z)', text, re.S):
        step, block = int(match[1]), match[2]
        if step in (50, 100, 150) or step > 165:
            continue
        for j, chunk in enumerate(block.split('Solution string: ')[1:]):
            if '<|im_start|>assistant\n' not in chunk:
                continue
            prompt, rest = chunk.split('<|im_start|>assistant\n', 1)
            question = prompt.split('Question: ', 1)[-1].split('<|im_end|>')[0].strip()
            source = qlookup.get(question)
            if source is None:
                continue
            response = re.split(r'\n(?:-{20,}|step:\d+|ACTIVE_TRAJ_NUM:|\[validation\])', rest)[0]
            if '<|im_end|>' in response:
                response = response.split('<|im_end|>', 1)[0] + '<|im_end|>'
            gold = list(source['reward_model']['ground_truth']['target'])
            printed.append(dict(method='dgpo', step=step, index=source['extra_info']['index'],
                question=question, gold=gold, decoded_response=response, source_file=str(DGPO.relative_to(ROOT)),
                **features(response, gold)))
    for lo, hi in [(1, 10), (61, 70), (151, 165)]:
        rs = [r for r in printed if lo <= r['step'] <= hi]
        if rs:
            summary['dgpo_log_coverage'][f'{lo}-{hi}'] = stats(rs)
    late = [r for r in printed if 151 <= r['step'] <= 165 and not r['score'] and not r['missing_answer']]
    for group, candidates in [('all_wrong', late), ('hit_wrong', [r for r in late if r['bounded_gold_hit']])]:
        review.extend(dict(review_group=group, **row) for row in sample(candidates, group, 6))
    for i, row in enumerate(review, 1):
        row['review_id'] = i
    # Inspect saved teacher/student scores, including tokens overlapping answer tags.
    # Coarse segments='answer' can omit the first content character (e.g. '>C').
    signal_rows = []
    seen = set()
    for case in review:
        identity = (case['method'], case['step'], case['index'])
        if case['method'] == 'dgpo' or identity in seen:
            continue
        seen.add(identity)
        with gzip.open(ROOT / case['source_file'], 'rt') as f:
            next(f)
            record = next(r for line in f for r in [json.loads(line)] if r['index'] == case['index'])
        joined = ''.join(record['token_texts'])
        matches = list(re.finditer(r'<answer>(.*?)</answer>', joined, re.S))
        if not matches:
            continue
        match = matches[-1]
        start, end = match.span(1)
        while start < end and joined[start].isspace():
            start += 1
        while end > start and joined[end - 1].isspace():
            end -= 1
        pos, tokens = 0, []
        for i, token in enumerate(record['token_texts']):
            nxt = pos + len(token)
            if pos < end and nxt > start and record['loss_mask'][i]:
                tokens.append(dict(token=token, token_index=i, segment=record['segments'][i],
                    teacher_logp=record['teacher_log_prob'][i], student_logp=record['student_log_prob'][i],
                    advantage=record['opd_advantage'][i], effective_coef=record['opd_effective_distillation_coef'][i]))
            pos = nxt
        signal_rows.append(dict(review_id=case['review_id'], method=case['method'], step=case['step'],
            index=case['index'], answer=case['answer'], tokens=tokens,
            decoded_answer_matches=joined[start:end].strip() == case['answer']))
    (OUT / 'review_token_signals.json').write_text(json.dumps(signal_rows, ensure_ascii=False, indent=2))
    (OUT / 'dgpo_printed_trajectories.json').write_text(json.dumps(printed, ensure_ascii=False, indent=2))
    (OUT / 'review_sample.json').write_text(json.dumps(review, ensure_ascii=False, indent=2))
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
