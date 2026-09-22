"""Question-paired analysis of the frozen factorial teacher experiment."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/evidence_history_control_cpu_20260915'


def main():
    records = [json.loads(s) for s in (OUT / 'teacher_paths.jsonl').read_text().splitlines()]
    manifest = json.loads((OUT / 'inputs.frozen.json').read_text())
    ix = {(r['qid'], r['history'], r['condition'], r['action']): r for r in records}
    assert len(records) == len(ix) == 96
    assert all(r['device'] == 'cpu' and not r['cuda_initialized'] for r in records)
    states = []
    for c in manifest['cases']:
        a, s = [ix[c['qid'], c['history'], c['condition'], action] for action in ['answer', 'search']]
        row = {k: c[k] for k in ['qid', 'history', 'condition']}
        for alpha in manifest['alphas']:
            key = str(alpha)
            row['odds_alpha_' + key] = a['targets'][key]['log_probability'] - s['targets'][key]['log_probability']
            row['word_odds_alpha_' + key] = a['targets'][key]['token_log_probabilities'][1] - s['targets'][key]['token_log_probabilities'][1]
            # Common '<' token must have the same probability on both paths.
            assert abs(a['targets'][key]['token_log_probabilities'][0] - s['targets'][key]['token_log_probabilities'][0]) < 1e-3
        for tau in manifest['temperatures']:
            key = str(tau)
            row['odds_temperature_' + key] = a['temperatures'][key]['log_probability'] - s['temperatures'][key]['log_probability']
        row['delta'] = row['odds_alpha_1.5'] - row['odds_alpha_0']
        row['word_delta'] = row['word_odds_alpha_1.5'] - row['word_odds_alpha_0']
        row['search_to_answer'] = row['odds_alpha_0'] < 0 < row['odds_alpha_1.5']
        states.append(row)
    with (OUT / 'states.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(states[0])); w.writeheader(); w.writerows(states)
    sx = {(r['qid'], r['history'], r['condition']): r for r in states}
    contrasts = []
    rng = np.random.default_rng(2026091511)
    qids = manifest['selected_qids']
    boot = rng.integers(0, len(qids), size=(20000, len(qids)))
    for history in ['neutral', 'fixed_student']:
        for condition in ['both_hops', 'with_distractors']:
            vals = np.array([sx[q, history, condition]['delta'] - sx[q, history, 'first_hop']['delta'] for q in qids])
            low, high = np.quantile(vals[boot].mean(1), [.025, .975])
            contrasts.append(dict(history=history, contrast=condition + '_minus_first_hop', n=len(qids),
                                  mean=float(vals.mean()), median=float(np.median(vals)), positive=int((vals > 0).sum()),
                                  ci_low=float(low), ci_high=float(high), per_question=dict(zip(qids, vals.tolist()))))
    (OUT / 'paired_contrasts.json').write_text(json.dumps(contrasts, indent=2) + '\n')
    lines = ['# 固定推理历史的教师动作诊断', '',
             '8题×3证据条件×2固定历史，共48状态、96完整动作标签评分。CPU FP32；没有GPU、检索服务或训练。', '',
             '每个历史条件内部跨证据严格复用相同历史token。neutral是合成中性控制；fixed_student复用同一题OPD step100仅第一跳条件的真实推理。', '',
             '下面的变化是 log P(<answer>)/P(<search>) 的 ER−OPD 差值，不是准确率或覆盖所有输出形式的动作概率。', '',
             '|历史|证据|题数|平均变化|中位变化|正向题数|搜索→回答翻转|', '|---|---|---:|---:|---:|---:|---:|']
    for history in ['neutral', 'fixed_student']:
        for condition in ['first_hop', 'both_hops', 'with_distractors']:
            rs = [r for r in states if r['history'] == history and r['condition'] == condition]
            vals = [r['delta'] for r in rs]
            lines.append(f"|{history}|{condition}|{len(rs)}|{np.mean(vals):+.3f}|{np.median(vals):+.3f}|{sum(v > 0 for v in vals)}|{sum(r['search_to_answer'] for r in rs)}|")
    lines += ['', '|历史|配对证据对照|平均差异|95%题目bootstrap区间|正向题数|', '|---|---|---:|---|---:|']
    for r in contrasts:
        lines.append(f"|{r['history']}|{r['contrast']}|{r['mean']:+.3f}|[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]|{r['positive']}/{r['n']}|")
    lines += ['', '这些题在此前学生实验中使用过，仅与先前6题教师探针不重叠。小样本题目区间不是训练种子区间。',
              '完整标签和动作词结果同时保存在states.csv；温度0.5/1/2仅为预声明有限网格对照，不能排除所有温度策略。',
              '固定回答偏置会令上述配对delta差为0。显著偏离0可排除该简单局部模型，但不是训练收益或方法新颖性的证明。',
              '加入第二跳也改变内容与长度；干扰文档对照只能部分控制这一点。固定历史可能不适合新证据，neutral也属于人工边界。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    (OUT / 'status.json').write_text(json.dumps(dict(state='complete', states=48, paths=96, cuda_initialized=False), indent=2) + '\n')
    print('HISTORY_CONTROL_ANALYSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
