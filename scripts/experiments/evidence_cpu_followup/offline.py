"""Offline opportunity/cost accounting; no inference and no retrieval calls."""
import collections
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'reports/evidence_stopping_gpu5_20260915'
OUT=ROOT/'reports/evidence_cpu_followup_20260915'


def rows(name):return [json.loads(s) for s in (SOURCE/name).read_text().splitlines()]


def main():
    t=rows('trajectories.jsonl');f=rows('forced_answers.jsonl')
    tx={(r['label'],r['qid'],r['condition']):r for r in t}
    opportunities=[]
    for label in sorted({r['label'] for r in f}):
        for metric in ['em','alias_em']:
            count=collections.Counter();calls=collections.Counter()
            for r in f:
                if r['label']!=label or r['ordinary_action']!='search':continue
                if not r['available']:count['intervention_unavailable']+=1;continue
                tr=tx[label,r['qid'],'both_hops']
                direct=r['forced_'+metric];continued=tr['final_'+metric]
                category=('both_correct' if direct and continued else 'forced_only_correct' if direct
                    else 'continuation_only_correct' if continued else 'both_wrong')
                count[category]+=1;calls[category]+=tr['search_calls']
            for category in ['both_correct','forced_only_correct','continuation_only_correct','both_wrong','intervention_unavailable']:
                opportunities.append(dict(label=label,metric=metric,category=category,n=count[category],
                    observed_search_calls=calls[category]))
    cost=[]
    for step in [50,100]:
        a={q:r for (l,q,c),r in tx.items() if l==f'opd-step{step}' and c=='natural'}
        b={q:r for (l,q,c),r in tx.items() if l==f'er-alpha1.5-step{step}' and c=='natural'}
        groups=collections.defaultdict(list)
        for q in a:
            category='both_correct' if a[q]['final_em'] and b[q]['final_em'] else 'opd_only_correct' if a[q]['final_em'] else 'er_only_correct' if b[q]['final_em'] else 'both_wrong'
            groups[category].append((a[q],b[q]))
        for cat,rs in groups.items():
            cost.append(dict(step=step,category=cat,n=len(rs),opd_calls=sum(a['search_calls'] for a,b in rs),
                er_calls=sum(b['search_calls'] for a,b in rs),
                er_minus_opd_calls=sum(b['search_calls']-a['search_calls'] for a,b in rs)))
    for name,values in [('stopping_opportunities',opportunities),('natural_cost_decomposition',cost)]:
        (OUT/(name+'.json')).write_text(json.dumps(values,indent=2)+'\n')
        with (OUT/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    lines=['# 已有轨迹的 CPU 成本与回答机会核对','',
        '不调用模型或检索服务。下面只分解已观测结果，不构造可以部署的理想停止策略。',
        '“强制答对而继续答错”是固定前缀下的分支对比，不能说自然推理已经包含正确答案；原始 EM 与预声明别名计分并列。干预不可用单列。','',
        '|模型|计分|类别|题数|正常续写实际搜索次数|','|---|---|---|---:|---:|']
    for r in opportunities:
        if r['label']=='initial-step0':continue
        lines.append(f"|{r['label']}|{r['metric']}|{r['category']}|{r['n']}|{r['observed_search_calls']}|")
    lines+=['','## 自然题：成本减少发生在哪些题上？','',
        '|步数|配对正确性类别|题数|OPD 搜索总数|ER 搜索总数|ER−OPD|','|---|---|---:|---:|---:|---:|']
    for r in cost:lines.append(f"|{r['step']}|{r['category']}|{r['n']}|{r['opd_calls']}|{r['er_calls']}|{r['er_minus_opd_calls']:+d}|")
    lines+=['','各类别按结果划分，只用于描述，不作为预先可识别的策略分组。后续 CPU 统一精度强制回答完成后，将使用新结果重新核对，保持旧表可追溯。']
    (OUT/'OFFLINE_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('OFFLINE_COMPLETE',len(opportunities),len(cost))


if __name__=='__main__':main()
