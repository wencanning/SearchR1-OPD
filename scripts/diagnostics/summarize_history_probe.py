"""Summarize every pre-registered history-probe score without choosing a metric."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/history_intervention_cpu_20260914'
CONDITIONS = ['original', 'hide_post_think', 'hide_pre_think_control', 'delete_post_think']
METRICS = ['full_sum', 'full_mean', 'answer_sum', 'answer_mean']


def main():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    rows = [json.loads(l) for l in (OUT / 'scores.jsonl').read_text().splitlines()]
    excluded_path = OUT / 'excluded.jsonl'
    excluded = [json.loads(l) for l in excluded_path.read_text().splitlines()] if excluded_path.exists() else []
    assert len(rows) == len({r['key'] for r in rows})
    assert len(rows) + len(excluded) == len(manifest['cases']), 'Experiment incomplete'
    flat, counts = [], {}
    for r in rows:
        assert set(r['scores']) == set(CONDITIONS)
        assert r['baseline_error'] < .5
        for c, v in r['scores'].items():
            # Both candidates must have the same prefix-only prediction.
            assert v['gold']['first_top_token'] == v['wrong']['first_top_token']
            assert abs(v['gold']['first_top_logp'] - v['wrong']['first_top_logp']) < 1e-3
            flat.append(dict(key=r['key'], group=r['group'], condition=c,
                first_top_token=v['gold']['first_top_token'],
                **{k:v['gold'][k]-v['wrong'][k] for k in METRICS}))
    with (OUT / 'margins.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    for group in sorted({r['group'] for r in rows}):
        counts[group] = {}
        for c in CONDITIONS:
            selected = [r for r in flat if r['group']==group and r['condition']==c]
            counts[group][c] = dict(n=len(selected), **{k:sum(r[k]>0 for r in selected) for k in METRICS})
    text = ['# 原始指标全口径汇总', '',
            '正值表示正确候选分数较高，负值表示错误候选较高；单位nat。不是自由生成准确率。', '',
            'original=原始；hide_post_think=隐藏搜索后推理；hide_pre_think_control=隐藏搜索前复述；delete_post_think=删除搜索后推理。', '']
    for metric in METRICS:
        text += [f'## {metric}', '', '|样本|'+'|'.join(CONDITIONS)+'|', '|---|'+'---:|'*4]
        for r in rows:
            values=[next(x for x in flat if x['key']==r['key'] and x['condition']==c)[metric] for c in CONDITIONS]
            text.append('|'+r['key']+'|'+'|'.join(f'{x:.4f}' for x in values)+'|')
        text.append('')
    text += ['## 复现与排除', '', f'完成{len(rows)}例，排除{len(excluded)}例。',
             f'原始首token复现最大误差：{max(r["baseline_error"] for r in rows):.6f}nat。',
             '每个条件的两个候选具有相同的前缀首token argmax，首token最高log概率差均小于0.001nat。',
             '', '## 排序计数（仅预选二候选比较）', '', '```json', json.dumps(counts,ensure_ascii=False,indent=2), '```']
    (OUT / 'RESULTS.md').write_text('\n'.join(text)+'\n')
    (OUT / 'rank_counts.json').write_text(json.dumps(counts, ensure_ascii=False, indent=2))
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
