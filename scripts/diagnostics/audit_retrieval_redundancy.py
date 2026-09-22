"""Read-only, exploratory audit of existing natural rollouts; no model or HTTP."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'reports/evidence_stopping_gpu5_20260915'
OUT = ROOT / 'reports/retrieval_redundancy_cpu_20260916'
METRICS = ['em', 'calls', 'repeat_query', 'repeat_observation', 'no_new_document', 'new_document_calls']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [SOURCE / 'trajectories.jsonl', SOURCE / 'retrieval_cache.json', Path(__file__)]
    protocol = dict(source_sha256={str(p): sha(p) for p in inputs}, condition='natural',
        population='All five checkpoints, all 64 previously inspected natural questions; no outcome selection',
        repeat_query='Exact query string occurred earlier in the same trajectory',
        repeat_observation='Identical observation token sequence occurred earlier in the same trajectory',
        no_new_document='All returned document content SHA256 hashes already appeared in earlier retrievals',
        new_document_calls='At least one returned document content hash was not seen in earlier retrievals',
        document_scope='Full cached returned documents, not necessarily the truncated visible text; not semantic usefulness',
        paired_bootstrap='5000 resamples of 64 question IDs; steps 50/100 averaged within question for pooled results',
        ci='95% percentile intervals, descriptive exploratory, no multiple-comparison correction', seed=2026091611,
        no_model=True, no_http=True, training_authorized_by_result=False)
    frozen = OUT / 'protocol.frozen.json'
    if frozen.exists():
        assert json.loads(frozen.read_text()) == protocol
    else:
        frozen.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    cache = json.loads(inputs[1].read_text())
    records = [json.loads(line) for line in inputs[0].read_text().splitlines()]
    records = [r for r in records if r['condition'] == 'natural']
    assert len(records) == 320
    rows, search_rows = [], []
    for r in records:
        queries, observations, documents = set(), set(), set()
        counts = collections.Counter()
        for turn, action in enumerate(r['actions']):
            if 'observation_ids' not in action:
                continue
            query = action['retrieval_query']
            assert query in cache, ('Missing cached factual query', query)
            docs = cache[query]
            hashes = {hashlib.sha256(d['document']['contents'].encode()).hexdigest() for d in docs}
            assert hashes, 'Empty factual retrieval must be audited separately'
            obs = tuple(action['observation_ids'])
            flags = dict(repeat_query=query in queries, repeat_observation=obs in observations,
                no_new_document=hashes <= documents, new_document_calls=bool(hashes - documents))
            counts.update({k: int(v) for k, v in flags.items()}); counts['calls'] += 1
            search_rows.append(dict(label=r['label'], qid=r['qid'], turn=turn, query=query,
                final_em=bool(r['final_em']), new_documents=len(hashes - documents), **flags))
            queries.add(query); observations.add(obs); documents.update(hashes)
        assert counts['calls'] == r['search_calls'], (r['label'], r['qid'], counts, r['search_calls'])
        assert counts['calls'] == counts['no_new_document'] + counts['new_document_calls']
        rows.append(dict(label=r['label'], qid=r['qid'], em=int(r['final_em']),
            **{k: counts[k] for k in METRICS if k != 'em'}))
    lookup = {(r['label'], r['qid']): r for r in rows}
    assert len(lookup) == 320
    labels = ['initial-step0', 'opd-step50', 'er-alpha1.5-step50', 'opd-step100', 'er-alpha1.5-step100']
    qids = sorted(r['qid'] for r in rows if r['label'] == 'initial-step0')
    assert len(set(qids)) == 64
    totals = {label: {k: sum(lookup[label, q][k] for q in qids) for k in METRICS} for label in labels}
    rng = np.random.default_rng(protocol['seed'])
    indices = rng.integers(0, 64, size=(5000, 64))
    contrasts = {}
    for steps in ([50], [100], [50, 100]):
        a = np.array([[np.mean([lookup[f'er-alpha1.5-step{s}', q][k] - lookup[f'opd-step{s}', q][k]
            for s in steps]) for k in METRICS] for q in qids])
        boots = a[indices].mean(axis=1)
        contrasts['+'.join(map(str, steps))] = {k: dict(delta=float(a[:, i].mean()),
            ci95=np.quantile(boots[:, i], [.025, .975]).tolist()) for i, k in enumerate(METRICS)}
    savings = collections.defaultdict(collections.Counter)
    for step in (50, 100):
        for q in qids:
            b, e = lookup[f'opd-step{step}', q], lookup[f'er-alpha1.5-step{step}', q]
            stratum = 'both_wrong' if not b['em'] and not e['em'] else 'other_outcomes'
            for k in METRICS[1:]:
                savings[stratum][k] += b[k] - e[k]
    (OUT / 'per_trajectory.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (OUT / 'per_search.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in search_rows))
    result = dict(state='complete', n_trajectories=len(rows), n_searches=len(search_rows),
        cache_coverage=1.0, totals=totals, paired_er_minus_opd=contrasts,
        descriptive_savings_opd_minus_er=dict(savings), no_gpu=True, no_http=True)
    (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 已有自然轨迹的检索重复性审计', '',
        '范围：同一旧探索面板64题×5个checkpoint；CPU离线读取，未生成新轨迹，未调用检索。所有实际检索均命中历史精确查询缓存。', '',
        '|Checkpoint|EM /64|搜索调用|重复查询调用|完全重复观察调用|未返回新文档的调用|返回新文档的调用|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label in labels:
        lines.append('|' + label + '|' + '|'.join(str(totals[label][k]) for k in METRICS) + '|')
    lines += ['', '## ER−OPD：每题差异（95%描述性区间）', '', '|Step|指标|差异|区间|', '|---|---|---:|---|']
    for step, vals in contrasts.items():
        for k, v in vals.items():
            lines.append(f"|{step}|{k}|{v['delta']:+.5f}|[{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]|")
    lines += ['', '## 解释边界', '',
        '1. “新文档”只指完整缓存正文的身份首次出现，不代表可见正文有新信息，更不代表答案所需证据。',
        '2. 重复查询、重复观察和无新文档三个指标有重叠，不能相加。只有“无新文档/有新文档”互斥且穷尽实际调用。',
        '3. 所有方法按全部64题计算调用数，避免只对搜索成功题进行筛选。双方都错的分层仅为结果后描述，不能作为因果子组。',
        '4. 历史配置/种子非严格匹配，面板已多次查看；当前差异不证明训练因果效果或可泛化收益。',
        '5. 两个step不是128个独立问题；汇总先对同题两step取均值，再按64题bootstrap。多指标区间未经多重比较校正。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
