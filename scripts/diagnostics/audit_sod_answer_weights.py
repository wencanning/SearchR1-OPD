"""CPU-only descriptive audit of saved SOD-without-GRPO signals, not full SOD."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/sod_answer_weights_cpu_20260916'
SOURCE = ROOT / 'verl_checkpoints/pure-sod-05B-n1-20260914-1220-coef1/opd_diagnostics'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [SOURCE / f'step_{i:06d}.jsonl.gz' for i in range(61, 71)]
    protocol = dict(sources={str(p): sha(p) for p in paths + [Path(__file__)]},
        population='All 1280 recorded HotpotQA training rollouts at steps 61-70; previously inspected window',
        baseline='Historical SOD step reweighting without GRPO, not the complete paper method',
        primary='Final answer step weight and first non-whitespace answer-content token signal by final EM',
        sign_threshold=.01, boundary='Exclude first content token if it overlaps opening/closing tag',
        no_teacher_correctness_labels=True, exploratory=True, gpu=False, http=False)
    frozen = OUT / 'protocol.frozen.json'
    if frozen.exists():
        assert json.loads(frozen.read_text()) == protocol
    else:
        frozen.write_text(json.dumps(protocol, indent=2) + '\n')
    rows = []
    for p in paths:
        with gzip.open(p, 'rt') as f:
            metadata = json.loads(next(f))
            assert metadata['selected_sequences'] == metadata['batch_size'] == 128
            count = 0
            for line in f:
                r = json.loads(line); count += 1
                assert r['data_source'] == 'hotpotqa'
                row = dict(step=r['global_step'], qid=r['index'], em=int(r['sequence_score']),
                    answer=r['final_answer'], gold=r['ground_truth_targets'])
                text = ''.join(r['token_texts'])
                matches = list(re.finditer(r'<answer>(.*?)</answer>', text, re.S))
                if not matches:
                    row['coverage'] = 'no_closed_answer'; rows.append(row); continue
                m = matches[-1]; start, end = m.span(1)
                while start < end and text[start].isspace(): start += 1
                if start == end:
                    row['coverage'] = 'empty_answer'; rows.append(row); continue
                offset = 0; found = False
                for i, token in enumerate(r['token_texts']):
                    nxt = offset + len(token)
                    if offset <= start < nxt:
                        found = True
                        row['first_token'] = token
                        if offset < m.start(1) or nxt > m.end(1) or not r['loss_mask'][i]:
                            row['coverage'] = 'boundary_or_mask'; break
                        # Reconstruct the contiguous policy step exactly as local implementation.
                        lo = i; hi = i + 1
                        while lo and r['loss_mask'][lo - 1]: lo -= 1
                        while hi < len(r['loss_mask']) and r['loss_mask'][hi]: hi += 1
                        d = statistics.mean(abs(x) for x in r['logp_gap_teacher_minus_student'][lo:hi])
                        gap = r['logp_gap_teacher_minus_student'][i]
                        weight = r['opd_effective_distillation_coef'][i]
                        assert abs(r['weighted_opd_advantage'][i] - weight * gap) < 1e-4
                        row.update(coverage='included', first_gap=gap, first_weight=weight,
                            answer_step_divergence=d, first_teacher_logp=r['teacher_log_prob'][i],
                            first_student_logp=r['student_log_prob'][i],
                            direction='positive' if gap > .01 else 'negative' if gap < -.01 else 'near_zero',
                            weight_group='up' if weight > 1.00001 else 'down' if weight < .99999 else 'one')
                        break
                    offset = nxt
                assert found
                rows.append(row)
            assert count == 128
    assert len(rows) == 1280
    cells = []
    for em in (0, 1):
        for direction in ('positive', 'negative', 'near_zero'):
            rs = [r for r in rows if r['coverage'] == 'included' and r['em'] == em and r['direction'] == direction]
            cells.append(dict(em=em, direction=direction, n=len(rs),
                weights=dict(collections.Counter(r['weight_group'] for r in rs)),
                mean_weight=statistics.mean(r['first_weight'] for r in rs) if rs else None,
                mean_gap=statistics.mean(r['first_gap'] for r in rs) if rs else None))
    summary = dict(state='complete', n=len(rows), correct=sum(r['em'] for r in rows),
        coverage=dict(collections.Counter(r['coverage'] for r in rows)), cells=cells,
        no_gpu=True, no_http=True, no_parameter_update=True)
    (OUT / 'rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# SOD历史权重与答案位置监督的CPU审计', '',
        '仅分析本地SOD无GRPO消融的step61–70全部1280条HotpotQA训练轨迹，不能代表完整SOD效果。',
        '答案字符边界重建保留首个非空白内容token；跨标签token排除。正负方向阈值±0.01，近零另列。', '',
        '|最终EM|首内容token教师−学生logp方向|轨迹数|升权|降权|权重=1|平均权重|',
        '|---|---|---:|---:|---:|---:|---:|']
    for c in cells:
        w = c['weights']
        mean = f"{c['mean_weight']:.4f}" if c['n'] else 'NA'
        lines.append(f"|{c['em']}|{c['direction']}|{c['n']}|{w.get('up', 0)}|{w.get('down', 0)}|{w.get('one', 0)}|{mean}|")
    lines += ['', 'Coverage: ' + json.dumps(summary['coverage']), '',
        '1. 最终答案EM错误不意味着其首token错误；正确答案和错误答案可能共享前缀。不能把正向首token信号计成整答案错误强化率。',
        '2. 正负logp差是局部采样优势，不是实际参数更新后概率变化。整步权重会同时放大或抑制正负方向。',
        '3. 该结果是训练轨迹上的描述性审计，不是教师正确率、因果比较或新方法准确率。EM还可能有别名假阴性。',
        '4. 未计算教师独立回答或正确候选概率，因此不能据此确认“师生共同犯错”的总体发生率。',
        '5. 任何改进先与完整SOD、普通OPD、仅加正确答案监督的简单对照比较，不以该表替代性能实验。', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
