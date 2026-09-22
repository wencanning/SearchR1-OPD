"""Post-hoc strict-role replication; retain the original four arms unchanged."""
import numpy as np
from common import ROOT,OUT,rows,save


def main():
    original=rows(ROOT/'reports/action_calibration_cpu_20260916/update_effects.jsonl')
    added=rows(OUT/'update_effects.jsonl')
    assert len(original)==4 and len(added)==2
    assert {r['variant'] for r in added}=={'strict_body_er','format_only_er'}
    before={(r['qid'],r['condition']):r for r in original[0]['before']}
    max_error=max(abs(x[k]-before[x['qid'],x['condition']][k]) for r in added for x in r['before']
        for k in ['log_odds']+(['reference_body_mean_logp'] if 'reference_body_mean_logp' in x else []))
    assert max_error<1e-6,('Original/new probe mismatch',max_error)
    effect={}
    for r in original+added:
        effect[r['variant']]={(x['qid'],x['condition']):{k:x[k]-before[x['qid'],x['condition']][k]
            for k in ['log_odds']+(['reference_body_mean_logp'] if 'reference_body_mean_logp' in x else [])} for x in r['after']}
    qids=sorted({q for q,c in before});assert len(qids)==8
    boot=np.random.default_rng(2026091620).integers(0,8,(20000,8))
    summary=[]
    for variant in effect:
        for condition in ['first_hop','both_hops','with_distractors']:
            for metric in ['log_odds']+(['reference_body_mean_logp'] if condition=='both_hops' else []):
                values=np.array([effect[variant][q,condition][metric] for q in qids])
                diff=values-np.array([effect['opd'][q,condition][metric] for q in qids])
                lo,hi=np.quantile(diff[boot].mean(1),[.025,.975])
                summary.append(dict(variant=variant,condition=condition,metric=metric,mean_update=float(values.mean()),
                    increment_over_opd=float(diff.mean()),ci95=[float(lo),float(hi)]))
    save(OUT/'summary.json',summary)
    lines=['# 严格正文/纯格式ER：CPU补充定位','',
        '事后追加两个臂，复用同一更新批次及8题探针，不是独立确认实验。GRPO=0、fresh AdamW单步；不是最终EM。',
        f'原始/新运行before探针最大差：{max_error:.3g}。异常片段及正文/协议混合token恢复普通OPD。','',
        '|变体|条件|指标|更新量|相对OPD增量|95%题目区间|', '|---|---|---|---:|---:|---|']
    for r in summary:
        lines.append(f"|{r['variant']}|{r['condition']}|{r['metric']}|{r['mean_update']:+.5f}|{r['increment_over_opd']:+.5f}|[{r['ci95'][0]:+.5f},{r['ci95'][1]:+.5f}]|")
    lines += ['','不能把不显著当作等效；参考答案logp提升不保证生成准确率提升。共享参数和非线性优化使分组变化不能线性相加。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(OUT/'status.json',dict(state='complete',variants=2,cuda_initialized=False,before_probe_max_error=max_error))
    print('STRICT_ROLE_CPU_COMPLETE',flush=True)


if __name__=='__main__':
    main()
