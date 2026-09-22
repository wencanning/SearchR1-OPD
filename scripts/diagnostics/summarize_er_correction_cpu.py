"""Recompute all report tables from frozen CPU experiment artifacts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reports/er_correction_cpu_20260914'
METHODS = ['observed','hidden','er_0_5','er_1','er_1_5','temperature_0_7','entropy_matched']
METRICS = ['full_sum','full_mean','answer_sum','answer_mean']


def read_jsonl(name):
    p=OUT/name
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def wilson(k,n):
    if not n:return None
    z=1.95996398454;p=k/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [center-radius,center+radius]


def main():
    torch.set_num_threads(1)
    manifest=json.loads((OUT/'probe_manifest.json').read_text())
    sample=json.loads((OUT/'sample_manifest.json').read_text())
    ann={r['review_id']:r for r in json.loads((OUT/'annotations.json').read_text())['labels']}
    cases={c['key']:c for c in manifest['cases']}
    allrows=read_jsonl('scores.jsonl');rows=[r for r in allrows if not r['excluded']]
    assert len({(r['key'],r['boundary']) for r in allrows})==len(allrows)
    annotations=[dict(review_id=c['review_id'],method=c['method'],step=c['step'],index=c['index'],
                      em=c['score'],**{k:v for k,v in ann[c['review_id']].items() if k!='review_id'}) for c in sample['cases']]
    pd.DataFrame(annotations).to_csv(OUT/'semantic_audit.csv',index=False)
    margins=[];local=[];signals=[];checks=[]
    for r in rows:
        c=cases[r['key']]
        checks.append(dict(review_id=r['review_id'],boundary=r['boundary'],**r['checks'],
            max_entropy_error=max(s['entropy_match_details']['max_entropy_error'] for s in r['scores'].values())))
        if c['group']!='correct_control':
            for method in METHODS:
                for metric in METRICS:
                    margins.append(dict(review_id=c['review_id'],key=c['key'],group=c['group'],boundary=r['boundary'],
                        method=method,metric=metric,value=r['scores']['gold'][method][metric]-r['scores']['original'][method][metric]))
            local.append(dict(review_id=c['review_id'],key=c['key'],group=c['group'],boundary=r['boundary'],
                gold_token=r['divergence']['gold_token'],wrong_token=r['divergence']['wrong_token'],
                **r['divergence']['log_odds']))
        if r['boundary']=='native':
            cache=OUT/'logits'/f"{c['review_id']}_native_original.pt"
            if not cache.exists() and OUT.name=='precision_sensitivity':
                cache=OUT.parent/'logits'/cache.name
            content=torch.load(cache,map_location='cpu',weights_only=True)['content']
            for method in METHODS:
                vals=np.array(r['scores']['original'][method]['token_logps'])
                old=np.array(c['saved_student_logps'])
                assert len(vals)==len(old)
                signals.append(dict(review_id=c['review_id'],group=c['group'],method=method,
                    full_advantage_sum=float((vals-old).sum()),positive_tokens=int((vals-old>0).sum()),n_tokens=len(vals),
                    answer_advantage_sum=float((vals-old)[content].sum()),
                    positive_answer_tokens=int(((vals-old)[content]>0).sum()),n_answer_tokens=len(content),
                    saved_answer_advantage_sum=float(np.array(c['saved_opd_advantage'])[content].sum()),
                    saved_weighted_answer_advantage_sum=float(np.array(c['saved_weighted_advantage'])[content].sum()),
                    original_saved_advantage_sum=sum(c['saved_opd_advantage']),
                    original_saved_weighted_advantage_sum=sum(c['saved_weighted_advantage']),
                    saved_advantage_identity_max_error=max(abs(t-s-a) for t,s,a in zip(c['saved_teacher_logps'],c['saved_student_logps'],c['saved_opd_advantage']))))
    mf=pd.DataFrame(margins);lf=pd.DataFrame(local)
    mf.to_csv(OUT/'candidate_margins.csv',index=False)
    lf.to_csv(OUT/'first_divergence.csv',index=False)
    pd.DataFrame(signals).to_csv(OUT/'sampled_advantages.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'numeric_checks.csv',index=False)
    stats={}
    for group in ['sufficient_failure','insufficient_control']:
        part=lf[(lf.group==group)&(lf.boundary=='native')] if not lf.empty else lf
        group_stats={}
        for method in METHODS:
            if len(part)==0:continue
            delta=part[method]-part.observed
            success=((part.observed<0)&(part[method]>0)).sum()
            harm=((part.observed>0)&(part[method]<0)).sum()
            n_wrong=int((part.observed<0).sum())
            group_stats[method]=dict(n=len(part),local_correct=int((part[method]>0).sum()),
                improved=int((delta>1e-4).sum()),worsened=int((delta < -1e-4).sum()),
                improved_over_0_1=int((delta>0.1).sum()),worsened_over_0_1=int((delta < -0.1).sum()),
                rescued=int(success),harmed=int(harm),observed_wrong=n_wrong,
                rescue_wilson=wilson(int(success),n_wrong),median_delta=float(delta.median()),
                median_delta_vs_entropy_match=float((part[method]-part.entropy_matched).median()))
        stats[group]=group_stats
    summary=dict(completed=(OUT/'COMPLETED.json').exists(),n_review=len(annotations),
        scope='CPU FP32 internal sensitivity; historical FP16 parity not required' if OUT.name=='precision_sensitivity' else 'Historical log-probability reproduction gate applied',
        historical_reproduction_failures=[r['review_id'] for r in rows if r['boundary']=='native' and r['checks'].get('historical_reproduction_pass') is False],
        annotation_counts={str(em):dict(Counter(a['semantic'] for a in annotations if a['em']==em)) for em in [0,1]},
        sufficient_semantic_failures=sum(a['semantic']=='wrong' and a['evidence']=='sufficient' for a in annotations),
        n_native=sum(r['boundary']=='native' for r in rows),n_early=sum(r['boundary']=='early' for r in rows),
        technical_exclusions=manifest['technical_exclusions'],numerical_exclusions=[r for r in allrows if r['excluded']],
        local_statistics=stats,teacher_forward_scoring_seconds=sum(r['seconds'] for r in allrows),
        generation=read_jsonl('generation.jsonl'),max_native_logp_error=max((r.get('native_logp_max_error',0) for r in checks),default=None),
        max_entropy_match_error=max((r['max_entropy_error'] for r in checks),default=None))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    lines=['# CPU ER 纠错诊断：机器汇总',f"\n状态：{'完成' if summary['completed'] else '运行中，结果不完整'}。",
           f"\n范围：{summary['scope']}。历史复核失败但保留用于FP32内部补充的案例：{summary['historical_reproduction_failures']}。",
           '\n## 语义审查',f"\n40条固定样本：{summary['annotation_counts']}。证据充分且语义错：{summary['sufficient_semantic_failures']}条。",
           '\n## 核心案例：第一个不同token的正确/错误候选log odds',
           '\n正值仅表示在该局部token处偏向正确候选，不能替代完整答案或自由生成。\n',
           '|Case|Gold token / original token|Observed|Hidden|ER .5|ER 1|ER 1.5|Temp .7|Entropy matched|',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in local:
        if r['group']=='sufficient_failure' and r['boundary']=='native':
            lines.append('|'+r['review_id']+'|'+repr(r['gold_token'])+' / '+repr(r['wrong_token'])+'|'+'|'.join(f"{r[m]:.4f}" for m in METHODS)+'|')
    lines+=['\n## 所有案例、全部完整答案计分口径','\n任何选择有利口径的结论都不成立。完整CSV：candidate_margins.csv。']
    for r in rows:
        if r['group']=='correct_control':continue
        lines += [f"\n### {r['review_id']} / {r['boundary']} / {r['group']}",
            '\n|Metric|'+'|'.join(METHODS)+'|','|---|'+'|'.join(['---:']*len(METHODS))+'|']
        for metric in METRICS:
            values=[r['scores']['gold'][m][metric]-r['scores']['original'][m][metric] for m in METHODS]
            lines.append('|'+metric+'|'+'|'.join(f'{v:.4f}' for v in values)+'|')
    lines+=['\n## 正确对照：正确答案正文log概率','\n|Case|Observed|ER1|Entropy matched|','|---|---:|---:|---:|']
    for r in rows:
        if r['group']=='correct_control':
            lines.append('|'+r['review_id']+'|'+'|'.join(f"{r['scores']['gold'][m]['answer_sum']:.4f}" for m in ['observed','er_1','entropy_matched'])+'|')
    lines+=['\n## 预先选定的答案边界自由续写','\n|Case|Target|Text|Stop|Reference EM|','|---|---|---|---|---|']
    for r in summary['generation']:
        lines.append('|'+r['review_id']+'|'+r['target']+'|'+r['text'].replace('\n',' ').replace('|',' / ')+'|'+r['stop_reason']+'|'+str(r['reference_em'])+'|')
    lines+=['\n## 数值复核',f"\n原生全答案最大逐token误差：{summary['max_native_logp_error']} nat；最大熵匹配误差：{summary['max_entropy_match_error']} nat。",
        '\n完整排除项、每个分布及sampled优势见JSON/CSV；源logits保存在logits/。',
        '\n## 范围','\n这是冻结教师目标诊断，无学生新训练，无GPU，无新的检索。来源SOD为无GRPO消融。',
        '\n边界敏感性额外删除最后一段生成历史并添加answer cue，不能解读为纯历史因果干预。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['generation','technical_exclusions','numerical_exclusions']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
