"""Qualify actual recovery targets without assuming teacher superiority."""
import collections
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'reports/recovery_queue_20260917'


def main():
    inp=json.loads((OUT/'train_inputs.frozen.json').read_text());cases={c['qid']:c for c in inp['cases']}
    values={}
    for model in ('student','teacher'):
        p=OUT/('continue_'+model);assert json.loads((p/'status.json').read_text())['state']=='complete'
        rows=[json.loads(l) for l in (p/'continuations.jsonl').read_text().splitlines()];keys={(r['qid'],r['replicate']) for r in rows};assert len(keys)==len(rows)
        values[model]=rows
    states=[json.loads(l) for l in (OUT/'collected/states.jsonl').read_text().splitlines()];eligible=[r for r in states if r['eligible'] and r['split']=='train']
    per=[];targets=[];student_targets=[]
    for state in eligible:
        q=state['qid'];ss=[r for r in values['student'] if r['qid']==q];ts=[r for r in values['teacher'] if r['qid']==q]
        assert len(ss)==len(ts)==inp['replicates']
        sc=sum(r['em'] for r in ss);tc=sum(r['em'] for r in ts)
        per.append(dict(qid=q,student_correct=sc,teacher_correct=tc,replicates=len(ss),student_all_failed=sc==0,teacher_rescues=sc==0 and tc>0,
            student_stop_reasons=[r['stop_reason'] for r in ss],teacher_stop_reasons=[r['stop_reason'] for r in ts],
            teacher_success_calls=[r['search_calls'] for r in ts if r['em']],teacher_success_tokens=[r['generated_tokens'] for r in ts if r['em']]))
        # Target candidate acquisition is not restricted to sampled-all-failure states.
        # Both observed failure and teacher success are required for the primary repair set.
        if sc==0:
            for t in ts:
                if t['em']:targets.append(t)
        for s in ss:
            if s['em']:student_targets.append(s)
    d=np.array([(r['teacher_correct']-r['student_correct'])/r['replicates'] for r in per]);rng=np.random.default_rng(2026091707);ci=np.quantile(d[rng.integers(0,len(d),size=(20000,len(d)))].mean(1),[.025,.975]).tolist() if len(d) else None
    def arm(rs):return dict(n=len(rs),correct=sum(r['em'] for r in rs),search_calls=sum(r['search_calls'] for r in rs),generated_tokens=sum(r['generated_tokens'] for r in rs),stop_reasons=dict(collections.Counter(r['stop_reason'] for r in rs)))
    result=dict(n_input_train=sum(c['split']=='train' for c in inp['cases']),n_eligible_train=len(per),n_input_eval=sum(c['split']=='eval' for c in inp['cases']),student_train=arm([r for r in values['student'] if r['split']=='train']),teacher_train=arm(values['teacher']),student_eval=arm([r for r in values['student'] if r['split']=='eval']),
      mean_teacher_student_delta=float(d.mean()) if len(d) else None,question_ci95=ci,observed_student_all_failed=sum(r['student_all_failed'] for r in per),teacher_rescued_questions=sum(r['teacher_rescues'] for r in per),
      verified_teacher_suffixes=len(targets),verified_student_suffixes=len(student_targets),per_question=per,
      successful_repair_suffixes_without_new_search=sum(r['search_calls']==0 for r in targets),successful_repair_suffixes_with_new_search=sum(r['search_calls']>0 for r in targets),
      qualification='Pilot. Four failed student samples do not establish impossibility. Teacher outcome verification does not prove student learnability, evidence grounding, or transfer. Train and eval question-disjoint. No oracle evidence used.')
    (OUT/'recovery_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'verified_teacher_train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in targets))
    (OUT/'verified_student_train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in student_targets))
    report=['# 新训练题同状态续写资格结果','',result['qualification'],'','|角色|正确/总数|新增搜索|生成token|','|---|---:|---:|---:|']
    for k in ('student_train','teacher_train','student_eval'):
        a=result[k];report.append(f"|{k}|{a['correct']}/{a['n']}|{a['search_calls']}|{a['generated_tokens']}|")
    report+=['',f"学生四次均失败：{result['observed_student_all_failed']}题；其中教师至少一次成功：{result['teacher_rescued_questions']}题；可用成功教师后缀：{len(targets)}条。",'资格只是目标是否存在；不能替代训练后新题生成。收集第一查询成本另计，不能宣称完整训练/交互预算已经匹配。']
    (OUT/'RECOVERY_REPORT.md').write_text('\n'.join(report)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
