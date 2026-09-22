"""Paired support-access contrasts; the three controls are not independent questions."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/evidence_support_access_cpu_20260916'


def main():
    manifest=json.loads((OUT/'inputs.frozen.json').read_text())
    records=[json.loads(s) for s in (OUT/'teacher_paths.jsonl').read_text().splitlines()]
    ix={(r['qid'],r['history'],r['action']):r for r in records}
    assert len(records)==len(ix)==32
    assert all(r['device']=='cpu' and not r['cuda_initialized'] for r in records)
    states=[]; contrasts=[]
    for c in manifest['cases']:
        a,s=[ix[c['qid'],c['history'],action] for action in ['answer','search']]
        for alpha in manifest['alphas']:
            key=str(alpha);ls={};words={}
            for name in a['variants']:
                av,sv=a['variants'][name][key],s['variants'][name][key]
                ls[name]=av['log_probability']-sv['log_probability']
                words[name]=av['token_log_probabilities'][1]-sv['token_log_probabilities'][1]
                assert abs(av['token_log_probabilities'][0]-sv['token_log_probabilities'][0])<1e-3
                states.append(dict(qid=c['qid'],history=c['history'],alpha=alpha,intervention=name,
                                   log_odds=ls[name],word_log_odds=words[name],masked_tokens=0 if name=='intact' else c['support_tokens']))
            controls=[ls[f'control_{i}_hidden'] for i in [1,2,3]]
            word_controls=[words[f'control_{i}_hidden'] for i in [1,2,3]]
            contrasts.append(dict(qid=c['qid'],history=c['history'],alpha=alpha,
                support_drop=ls['intact']-ls['support_hidden'],control_drop=ls['intact']-float(np.mean(controls)),
                specificity=float(np.mean(controls))-ls['support_hidden'],
                word_specificity=float(np.mean(word_controls))-words['support_hidden'],
                all_three_controls_above_support=all(x>ls['support_hidden'] for x in controls),
                support_answer_to_search=ls['intact']>0>ls['support_hidden']))
    for name,rs in [('states.csv',states),('contrasts.csv',contrasts)]:
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    rng=np.random.default_rng(2026091601);boot=rng.integers(0,8,size=(20000,8));summary=[]
    for history in ['neutral','fixed_student']:
        for alpha in manifest['alphas']:
            rs=[r for r in contrasts if r['history']==history and r['alpha']==alpha]
            vals=np.array([r['specificity'] for r in rs]);low,high=np.quantile(vals[boot].mean(1),[.025,.975])
            summary.append(dict(history=history,alpha=alpha,n=8,mean_specificity=float(vals.mean()),median_specificity=float(np.median(vals)),
                positive=int((vals>0).sum()),ci_low=float(low),ci_high=float(high),
                mean_support_drop=float(np.mean([r['support_drop'] for r in rs])),
                mean_control_drop=float(np.mean([r['control_drop'] for r in rs])),
                all_three_positive=sum(r['all_three_controls_above_support'] for r in rs)))
    # With identical hidden reference, single-token contrasts are algebraically scaled.
    cx={(r['qid'],r['history'],r['alpha']):r for r in contrasts}
    errors=[abs(r['word_specificity']-(1+r['alpha'])*cx[r['qid'],r['history'],0]['word_specificity']) for r in contrasts]
    (OUT/'ALGEBRA_CHECK.json').write_text(json.dumps(dict(max_single_word_scaling_error=max(errors),
       interpretation='Scaling is a construction identity, not independent evidence for ER effectiveness.'),indent=2)+'\n')
    assert max(errors)<1e-3,max(errors)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# 关键证据与等量非支持内容的访问对照','',
        '8道教师探针新题，2种固定历史。每题3个非支持对照在题内平均，不能当成24道独立问题。输入token、位置、历史、文档数量完全固定；只有直接访问的key列改变。', '',
        'K=屏蔽非支持内容后的平均回答/搜索log-odds − 屏蔽支持内容后的log-odds。K>0表示屏蔽支持内容更降低回答偏好。', '',
        '|历史|α|关键内容屏蔽的平均下降|非支持内容屏蔽的平均下降|K均值|K中位数|95%题目区间|K正向题数|',
        '|---|---:|---:|---:|---:|---:|---|---:|']
    for r in summary:
        lines.append(f"|{r['history']}|{r['alpha']}|{r['mean_support_drop']:+.3f}|{r['mean_control_drop']:+.3f}|{r['mean_specificity']:+.3f}|{r['median_specificity']:+.3f}|[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]|{r['positive']}/8|")
    primary=[r for r in summary if r['alpha'] in [0,1.5]]
    gate=all(r['mean_specificity']>0 and r['median_specificity']>0 and r['positive']>=6 for r in primary)
    lines += ['',f'预声明探索性继续门槛：{"通过" if gate else "未通过"}（两种历史、α0和1.5均须均值和中位数为正，至少6/8题正向）。',
        '通过只支持局部动作偏好对关键支持内容的访问更敏感；尚不能证明训练学到更好停止策略或提高准确率。未通过则不继续主张关键证据特异性。',
        '全隐藏参考固定时，单词层面对干预的ER响应等于(1+α)倍普通教师响应，这是代数恒等关系。完整标签含后续不同前缀的归一化，不强制完全同比例。放大本身不是实验创新证据。',
        '非支持内容指未被HotpotQA标注为支持事实且不含匹配答案别名的内容，不保证完全无用。等长度与连续区间长度匹配仍不能消除绝对位置、语义冗余和边界token的影响。',
        '人工attention屏蔽不是自然检索文档编辑，历史仍可能携带证据信息；小样本区间不包括训练种子方差。log-odds无有意义的相对百分比提升。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    (OUT/'status.json').write_text(json.dumps(dict(state='complete',paths=32,states=16,exploratory_gate_pass=gate,cuda_initialized=False),indent=2)+'\n')
    print('SUPPORT_ACCESS_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
