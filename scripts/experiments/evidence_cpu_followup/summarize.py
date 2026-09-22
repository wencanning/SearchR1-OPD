import collections
import csv
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/evidence_cpu_followup_20260915'


def rows(name):
    p=OUT/name
    return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []


def main():
    student=rows('student_answers.jsonl');teacher=rows('teacher_preferences.jsonl')
    assert len(student)==170 and len(teacher)==24
    assert len({(r['label'],r['qid']) for r in student})==170
    assert len({(r['qid'],r['condition']) for r in teacher})==24
    assert all(r['device']=='cpu' and not r['cuda_initialized'] for r in student+teacher)
    lines=['# CPU 后续诊断结果','',
        '全部前向在 CPU FP32 上执行；没有 CUDA 初始化，也没有检索服务请求。',
        '固定前缀来自上一轮 BF16 模型的既有推理。这轮重新生成全部170条回答，允许自然缺省推理，不与上一轮159条GPU强制回答拼接统计。','',
        '|模型|总题数|可执行强制回答|无推理段|原始EM正确数|预声明别名正确数|','|---|---:|---:|---:|---:|---:|']
    summary=[]
    for label in sorted({r['label'] for r in student}):
        rs=[r for r in student if r['label']==label]
        r=dict(label=label,n=len(rs),available=sum(x['available'] for x in rs),
            no_reasoning=sum(x['reasoning_present'] is False for x in rs),
            em=sum(x['em'] for x in rs),alias_em=sum(x['alias_em'] for x in rs))
        summary.append(r);lines.append(f"|{label}|{r['n']}|{r['available']}|{r['no_reasoning']}|{r['em']}|{r['alias_em']}|")
    ix={(r['label'],r['qid']):r for r in student};paired=[]
    qids=sorted(q for l,q in ix if l=='opd-step50')
    rng=np.random.default_rng(2026091510);boot=rng.integers(0,len(qids),size=(20000,len(qids)))
    for step_name,steps in [('50',[50]),('100',[100]),('mean50_100',[50,100])]:
        for metric in ['em','alias_em']:
            a=np.array([np.mean([ix[f'opd-step{s}',q][metric] for s in steps]) for q in qids])
            b=np.array([np.mean([ix[f'er-alpha1.5-step{s}',q][metric] for s in steps]) for q in qids])
            low,high=np.quantile((b-a)[boot].mean(1),[.025,.975])*100
            paired.append(dict(step=step_name,metric=metric,baseline=float(a.mean()),er=float(b.mean()),
                delta_pp=float((b-a).mean()*100),ci_low_pp=float(low),ci_high_pp=float(high)))
    (OUT/'student_paired.json').write_text(json.dumps(paired,indent=2)+'\n')
    lines+=['','## 教师动作词偏好：六个固定问题，每题四个证据条件','',
        '这里是保留同一baseline推理后，在公共 `<` 前缀下 answer/search 的条件偏好；不是完整动作概率或真实训练梯度。隐藏视图只切断正文直接访问，已有推理仍可携带证据信息。',
        '|证据条件|状态数|ER提高回答相对偏好的状态数|平均log-odds变化（α1.5）|observed偏搜索→ER偏回答|',
        '|---|---:|---:|---:|---:|']
    teacher_summary=[]
    for condition in ['first_hop','evidence_removed','with_distractors','both_hops']:
        rs=[r for r in teacher if r['condition']==condition]
        s=dict(condition=condition,n=len(rs),positive=sum(r['delta_answer_search_log_odds']>0 for r in rs),
            mean_delta=float(np.mean([r['delta_answer_search_log_odds'] for r in rs])),
            search_to_answer=sum(r['targets']['0']['answer_search_log_odds']<0<r['targets']['1.5']['answer_search_log_odds'] for r in rs))
        teacher_summary.append(s);lines.append(f"|{condition}|{s['n']}|{s['positive']}|{s['mean_delta']:+.4f}|{s['search_to_answer']}|")
    (OUT/'teacher_summary.json').write_text(json.dumps(teacher_summary,indent=2)+'\n')
    ti={(r['qid'],r['condition']):r for r in teacher};specific=[]
    for q in sorted({r['qid'] for r in teacher}):
        specific.append(dict(qid=q,sufficient_minus_first_hop=ti[q,'both_hops']['delta_answer_search_log_odds']-ti[q,'first_hop']['delta_answer_search_log_odds'],
            sufficient_minus_distractors=ti[q,'both_hops']['delta_answer_search_log_odds']-ti[q,'with_distractors']['delta_answer_search_log_odds']))
    (OUT/'teacher_state_specificity.json').write_text(json.dumps(specific,indent=2)+'\n')
    lines+=['',
        '解释门槛：若ER在充分和不足证据条件都同样推高回答偏好，只能支持一般动作偏移，不能声称更会判断何时结束。即使具有状态特异性，六题也仅为可行性诊断，不能证明训练因果或总体准确率收益。',
        '原始教师结果含全词表归一化、α=0/.5/1/1.5和α1.5的熵匹配温度对照。温度对照不匹配时保留缺失，不伪造。CPU强制回答的配对区间不包含训练种子方差。','']
    (OUT/'CPU_REPORT.md').write_text('\n'.join(lines))
    print('CPU_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
