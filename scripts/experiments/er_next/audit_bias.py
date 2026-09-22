"""Exploratory, CPU-only individual-state audit of already frozen predictions."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/er_next_20260916'
SOURCE = ROOT / 'reports/action_calibration_cpu_20260916/bias_test_details.csv'


def main():
    data = list(csv.DictReader(SOURCE.open()))
    results = []
    for step in (50, 100):
        for condition in ('first_hop', 'both_hops', 'with_distractors'):
            rows = [r for r in data if int(r['step']) == step and r['condition'] == condition]
            assert len(rows) == 32 and len({r['qid'] for r in rows}) == 32
            delta = np.array([float(r['er']) - float(r['opd_bias']) for r in rows])
            results.append(dict(step=step, condition=condition, n=32,
                mean_delta=float(delta.mean()), mean_absolute_error=float(abs(delta).mean()),
                max_absolute_error=float(abs(delta).max()),
                states_above_10pp=int((abs(delta) > .1).sum()),
                canonical_argmax_disagreements=sum((float(r['er']) >= .5) != (float(r['opd_bias']) >= .5) for r in rows)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'bias_individual_audit.json').write_text(json.dumps(dict(
        exploratory=True, source=str(SOURCE), sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        rows=results), indent=2) + '\n')
    lines = ['# 逐状态偏置审计（探索性CPU统计）', '',
        '这是对已查看测试结果的补充描述，不增加新的确认性检验。每行32题。', '',
        '|Step|证据条件|平均差(pp)|平均绝对差(pp)|最大绝对差(pp)|差>10pp题数|规范动作argmax不同题数|',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"|{r['step']}|{r['condition']}|{100*r['mean_delta']:.2f}|{100*r['mean_absolute_error']:.2f}|{100*r['max_absolute_error']:.2f}|{r['states_above_10pp']}|{r['canonical_argmax_disagreements']}|")
    lines += ['', '均值相近不能推出逐题相同。0.5阈值只用于两个规范开标签的条件argmax，不能等同于自由生成的实际动作。',
              '这些差异是否有用必须与独立分支收益相连；不能凭误差大小推断ER改善决策。']
    (OUT / 'BIAS_INDIVIDUAL_REPORT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
