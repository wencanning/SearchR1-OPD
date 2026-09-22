"""Question-paired descriptive analysis; this is a diagnostic, not policy training."""
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'reports/recovery_20260917'


def main():
    inp=json.loads((OUT/'probe_inputs.frozen.json').read_text());variants=['actual_native','actual_neutral','support_native','support_neutral'];models={};result={};rng=np.random.default_rng(2026091704)
    sample=rng.integers(0,len(inp['cases']),size=(20000,len(inp['cases'])))
    def ci(x):return dict(delta=float(x.mean()),ci95=np.quantile(x[sample].mean(1),[.025,.975]).tolist())
    for name in ('student','teacher'):
        p=OUT/('probe_'+name);status=json.loads((p/'status.json').read_text());assert status['state']=='complete'
        rows=[json.loads(l) for l in (p/'answers.jsonl').read_text().splitlines()];by={(r['qid'],r['variant'],r['replicate']):r for r in rows};expected={(c['qid'],v,i) for c in inp['cases'] for v in variants for i in range(inp['replicates'])};assert set(by)==expected and len(rows)==len(expected)
        values=np.array([[np.mean([by[c['qid'],v,i]['em'] for i in range(inp['replicates'])]) for v in variants] for c in inp['cases']]);models[name]=values
        result[name]=dict(n_questions=len(inp['cases']),n_generations=len(rows),counts={v:dict(correct=sum(r['em'] for r in rows if r['variant']==v),total=sum(r['variant']==v for r in rows),invalid=sum(not r['valid'] for r in rows if r['variant']==v)) for v in variants},
          remove_latest_think_actual=ci(values[:,1]-values[:,0]),support_with_native_history=ci(values[:,2]-values[:,0]),support_with_neutral_history=ci(values[:,3]-values[:,1]),
          remove_latest_think_support=ci(values[:,3]-values[:,2]),raw_sha256=hashlib.sha256((p/'answers.jsonl').read_bytes()).hexdigest())
    result['teacher_minus_student']={v:ci(models['teacher'][:,i]-models['student'][:,i]) for i,v in enumerate(variants)}
    result['qualification']='16/64 input-eligible previously inspected questions; support snippets privileged, forced answers not full rollouts, history removal also changes length; two samples per question. No effectiveness or novelty claim.'
    (OUT/'probe_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    report=['# 证据与最近解释的四格诊断','',result['qualification'],'','|条件|学生|教师|','|---|---:|---:|']
    for v in variants:
        a=result['student']['counts'][v];b=result['teacher']['counts'][v];report.append(f"|{v}|{a['correct']}/{a['total']} ({a['correct']/a['total']:.1%})|{b['correct']}/{b['total']} ({b['correct']/b['total']:.1%})|")
    report+=['','去除的是最后一次检索后的think正文；之前查询与历史保留。标注支持句≤512token且不截断，但不是已人工核定的完整推理链。样本按输入覆盖筛选，未按成功与失败选题。','详细配对区间见probe_summary.json。禁止把此强制回答诊断当作多轮恢复或训练收益。']
    (OUT/'PROBE_REPORT.md').write_text('\n'.join(report)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
