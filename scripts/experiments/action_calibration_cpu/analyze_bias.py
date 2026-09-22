"""Calibrate on 16 questions; test on 32 disjoint questions with frozen biases."""
import argparse
import csv
import numpy as np
from common import OUT,M,sha,rows,save,verify_inputs


def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-60,60)))


def calibrate():
    m=verify_inputs();data=rows(OUT/'scores_calibration.jsonl');assert len(data)==192
    ix={(r['label'],r['qid'],r['condition']):r for r in data};assert len(ix)==192
    biases={}
    for step in [50,100]:
        keys=[(c['qid'],c['condition']) for c in m['cases'] if c['split']=='calibration']
        opd=np.array([ix[f'opd-step{step}',*k]['log_odds'] for k in keys])
        er=np.array([ix[f'er-step{step}',*k]['log_odds'] for k in keys]);target=float(sigmoid(er).mean())
        lo,hi=-256.,256.
        assert sigmoid(opd+lo).mean()<=target<=sigmoid(opd+hi).mean()
        for _ in range(100):
            mid=(lo+hi)/2
            if sigmoid(opd+mid).mean()<target:lo=mid
            else:hi=mid
        b=(lo+hi)/2
        biases[str(step)]=dict(b=b,target_er_mean_probability=target,calibrated_opd_mean_probability=float(sigmoid(opd+b).mean()))
    frozen=dict(biases=biases,calibration_sha256=sha(OUT/'scores_calibration.jsonl'),input_sha256=sha(OUT/'inputs.frozen.json'),
                calibrated_only_on_split='calibration',probability='P(canonical answer path | canonical answer or search paths)')
    path=OUT/'bias.frozen.json'
    if path.exists():assert M.read(path)==frozen
    else:save(path,frozen)
    print('BIAS_FROZEN',biases,flush=True)


def summarize():
    m=verify_inputs();frozen=M.read(OUT/'bias.frozen.json')
    assert frozen['input_sha256']==sha(OUT/'inputs.frozen.json') and frozen['calibration_sha256']==sha(OUT/'scores_calibration.jsonl')
    rs=rows(OUT/'scores_test.jsonl');assert len(rs)==384
    ix={(r['label'],r['qid'],r['condition']):r for r in rs};assert len(ix)==384
    assert all(r['device']=='cpu' and not r['cuda_initialized'] for r in rs)
    qids=m['split_qids']['test'];conditions=['first_hop','both_hops','with_distractors']
    rng=np.random.default_rng(2026091603);boot=rng.integers(0,len(qids),size=(20000,len(qids)))
    summary=[];details=[];separation=[]
    for step in [50,100]:
        b=frozen['biases'][str(step)]['b'];by_condition={}
        for condition in conditions:
            a=np.array([ix[f'opd-step{step}',q,condition]['log_odds'] for q in qids])
            e=np.array([ix[f'er-step{step}',q,condition]['log_odds'] for q in qids])
            systems={'opd':sigmoid(a),'opd_bias':sigmoid(a+b),'er':sigmoid(e)};by_condition[condition]=systems
            delta=systems['er']-systems['opd_bias'];low,high=np.quantile(delta[boot].mean(1),[.025,.975])
            for system,p in systems.items():
                summary.append(dict(step=step,condition=condition,system=system,n=32,mean_answer_probability=float(p.mean()),
                    canonical_answer_choices=int((p>=.5).sum())))
            for i,q in enumerate(qids):details.append(dict(step=step,qid=q,condition=condition,opd=float(systems['opd'][i]),
                opd_bias=float(systems['opd_bias'][i]),er=float(systems['er'][i]),er_minus_bias=float(delta[i])))
            separation.append(dict(step=step,contrast=condition,delta=float(delta.mean()),ci_low=float(low),ci_high=float(high),
                within_predeclared_10pp_band=bool(low>=-.1 and high<=.1)))
        for system in ['opd','opd_bias','er']:
            d=by_condition['both_hops'][system]-(by_condition['first_hop'][system]+by_condition['with_distractors'][system])/2
            low,high=np.quantile(d[boot].mean(1),[.025,.975])
            separation.append(dict(step=step,contrast='sufficient_minus_mean_incomplete',system=system,delta=float(d.mean()),ci_low=float(low),ci_high=float(high)))
        d=(by_condition['both_hops']['er']-(by_condition['first_hop']['er']+by_condition['with_distractors']['er'])/2)-(
            by_condition['both_hops']['opd_bias']-(by_condition['first_hop']['opd_bias']+by_condition['with_distractors']['opd_bias'])/2)
        low,high=np.quantile(d[boot].mean(1),[.025,.975]);separation.append(dict(step=step,contrast='er_minus_bias_condition_separation',delta=float(d.mean()),ci_low=float(low),ci_high=float(high)))
    save(OUT/'bias_summary.json',summary);save(OUT/'bias_paired.json',separation)
    with (OUT/'bias_test_details.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    lines=['# 固定回答偏置对照：学生局部动作选择','',
        '16题校准，32题测试，按问题严格分离；step100为主、step50为预声明复核。共同中性历史、同一证据输入，CPU FP32，未调用检索服务。',
        '概率只在规范<answer>/<search>两条完整路径内归一化；不涵盖think或其他输出形式，不是自由生成或最终正确率。', '',
        '|检查点|证据条件|OPD回答倾向|OPD＋固定偏置|ER回答倾向|', '|---|---|---:|---:|---:|']
    for step in [50,100]:
        for condition in conditions:
            vals={r['system']:r['mean_answer_probability'] for r in summary if r['step']==step and r['condition']==condition}
            lines.append(f"|{step}|{condition}|{100*vals['opd']:.1f}%|{100*vals['opd_bias']:.1f}%|{100*vals['er']:.1f}%|")
    lines += ['', '|检查点|ER−偏置OPD对照|平均差(pp)|95%题目区间(pp)|','|---|---|---:|---|']
    for r in separation:
        if 'system' not in r:lines.append(f"|{r['step']}|{r['contrast']}|{100*r['delta']:+.2f}|[{100*r['ci_low']:+.2f},{100*r['ci_high']:+.2f}]|")
    lines += ['', 'b只使用校准集匹配总体条件回答概率，不按测试证据类型调节；并未强制匹配离散动作比例。',
        '±10pp带是预声明局部概率近似复现标准，不是任务准确率非劣效界限。若全部条件均落在该带内，只能说均值层面近似；不能说明逐题或完整任务等价。',
        '材料充分性是控制条件，不是最佳行动的oracle标签；不足材料下选择回答不能直接计作语义错误。',
        '正向条件分离差支持进一步测试状态相关性；区间含零不证明无差异。历史checkpoint比较仍有训练代码/种子混杂。','']
    (OUT/'BIAS_REPORT.md').write_text('\n'.join(lines));print('BIAS_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['calibrate','summarize']);a=parser.parse_args()
    try:calibrate() if a.stage=='calibrate' else summarize()
    except Exception as e:save(OUT/'failure.json',dict(stage='analyze_bias',error=repr(e)));raise
