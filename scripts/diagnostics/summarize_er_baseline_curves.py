"""Compare actual 0.5B runs at common evaluation steps; preserve all runs and metrics."""
import csv,json,statistics,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/er_baseline_curves_20260915'
COUNTS={'popqa':141,'2wikimultihopqa':125,'triviaqa':112,'hotpotqa':73,'nq':36,'musique':24,'bamboogle':1}
METRICS=['macro7','micro512','macro6','hotpotqa']
PRIMARY={'zd6ublab':'OPD + GRPO', 'iv3lo93u':'ER alpha=1', '4amvkkgl':'ER alpha=1.5 run A', '1k442z39':'ER alpha=1.5 run B'}

def dump(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def csvsave(name,rs):
 if not rs:return
 keys=list(dict.fromkeys(k for r in rs for k in r))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rs)
def get(c,path):
 for k in path.split('.'):
  if not isinstance(c,dict):return None
  c=c.get(k)
 return c
def flat(c,p=''):
 if not isinstance(c,dict):return {p:c}
 out={}
 for k,v in c.items():out.update(flat(v,p+'.'+k if p else k))
 return out

def main():
 runs=json.loads((OUT/'inventory.json').read_text());hist=json.loads((OUT/'histories.json').read_text())
 records=[];curves={};checks=[];summary=[]
 for r in runs:
  c=r['config'];curve={}
  for h in hist[r['local_id']]:
   if not all('val/test_score/'+d in h for d in COUNTS):continue
   if get(c,'data.val_files')!='data/nq_hotpotqa_train_30k_no_cold_start/validation_diagnostic_512.parquet':continue
   v={d:h['val/test_score/'+d] for d in COUNTS};macro=statistics.mean(v.values())
   if 'val/test_score/Avg' in h:
    error=abs(macro-h['val/test_score/Avg']);assert error<1e-7,(r['run'],error)
   # Scores should correspond to integer correct counts on the current fixed split.
   errs={d:abs(v[d]*COUNTS[d]-round(v[d]*COUNTS[d])) for d in COUNTS}
   assert max(errs.values())<1e-4,(r['run'],errs)
   row=dict(run=r['run'],local_id=r['local_id'],name=r['name'],step=h['training_step'],
       macro7=macro,micro512=sum(v[d]*COUNTS[d] for d in COUNTS)/512,
       macro6=statistics.mean(v[d] for d in COUNTS if d!='bamboogle'),hotpotqa=v['hotpotqa'],
       bamboogle=v['bamboogle'],correct512=round(sum(v[d]*COUNTS[d] for d in COUNTS)),
       hotpot_correct=round(v['hotpotqa']*73),**{'score_'+d:v[d] for d in COUNTS})
   assert row['step'] not in curve,('duplicate evaluation step',r['run'],row['step'])
   curve[row['step']]=row;records.append(row)
  curves[r['local_id']]=curve
  if curve:
   row=dict(run=r['run'],name=r['name'],local_id=r['local_id'],evaluations=len(curve),steps=sorted(curve),
       train_source=get(c,'data.train_data_source'),student=get(c,'actor_rollout_ref.model.path'),
       lambda_distill=get(c,'algorithm.opd.lambda_distill'),alpha=get(c,'algorithm.opd.evidence_residual_alpha'),
       n_agent=get(c,'actor_rollout_ref.rollout.n_agent'),grpo_coef=get(c,'algorithm.opd.grpo_reward_coef'))
   for m in METRICS:
    row[m+'_all_available_mean']=statistics.mean(v[m] for v in curve.values())
    row[m+'_last']=curve[max(curve)][m];row[m+'_best']=max(v[m] for v in curve.values())
   summary.append(row)
 byid={r['run']:r for r in runs if get(r['config'],'actor_rollout_ref.model.path')=='data/student/0.5B'}
 base=byid['zd6ublab'];bc=curves[base['local_id']];common=[50,100]
 comparisons=[];diffs={}
 for rid,label in PRIMARY.items():
  r=byid[rid];curve=curves[r['local_id']];assert all(s in curve for s in common)
  item=dict(run=rid,label=label,steps=common)
  for m in METRICS:
   value=statistics.mean(curve[s][m] for s in common);b=statistics.mean(bc[s][m] for s in common)
   item[m]=value;item[m+'_delta_pp']=100*(value-b);item[m+'_relative_change_pct']=100*(value/b-1)
  comparisons.append(item)
  x=flat(base['config']);y=flat(r['config'])
  diffs[rid]={k:dict(baseline=x.get(k),run=y.get(k)) for k in x.keys()|y.keys() if x.get(k)!=y.get(k) and not k.startswith('_wandb')}
 # Same hyperparameters, but not known independent random seeds: describe run dispersion only.
 repeated=[x for x in comparisons if x['run'] in ['4amvkkgl','1k442z39']]
 dispersion={m:dict(mean=statistics.mean(x[m] for x in repeated),sample_std=statistics.stdev(x[m] for x in repeated),n_runs=2) for m in METRICS}
 csvsave('all_validation_points.csv',records);csvsave('all_runs_summary.csv',summary);csvsave('matched_comparison.csv',comparisons)
 dump('config_differences.json',diffs)
 result=dict(as_of=datetime.datetime.now().astimezone().isoformat(),counts=COUNTS,primary_run_ids=PRIMARY,
     common_steps=common,comparisons=comparisons,repeat_dispersion_not_seed_CI=dispersion,
     current_run_real_alpha=get(byid['1k442z39']['config'],'algorithm.opd.evidence_residual_alpha'),
     no_gpu_experiments_launched=True,macro_recompute_and_integer_count_checks='passed')
 dump('comparison.json',result)
 report=['# 0.5B baseline vs ER-OPD：已有训练曲线复核','',
 '结论：现有最接近配置的比较没有显示稳定的整体准确率增益。ER α=1 的七数据集宏平均看似提升较大，但主要来自只有1题的Bamboogle；按512道题加权后，50/100步曲线均值反而略降。α=1.5两次运行的总体曲线均值与baseline接近。这个结果不足以宣判方法无效，也不支持稳定有效的结论。','',
 '## 比较口径','',
 '- 数据取自本地W&B二进制历史的完整精度记录，不使用界面平滑线或挑选最佳点；histories.json保留读取快照，all_validation_points.csv保留逐点数值。',
 '- 主baseline为zd6ublab：0.5B OPD+GRPO，λ=.01、GRPO系数1、每题8rollouts、train batch128、同一初始学生与7B教师、HotpotQA训练；三个主要ER运行对应相同这些配置。',
 '- 共同验证点只有50、100步。所谓“曲线平均”是这两个点的算术平均，不包含未测step0，不推测150/200步；两个等间隔端点的归一化梯形面积也恰好等于此均值。',
 '- 原Avg是7个验证数据集的宏平均。micro512是512题总体EM；macro6是不含Bamboogle的6集宏平均，作为事后敏感性分析，不取代原定指标；HotpotQA含73题。',
 '- 两次alpha1.5不是已验证独立seed实验，不能用训练步作为独立样本计算显著性。baseline只有一个λ=.01运行。','',
 '## 最接近配置：共同50/100步','',
 '|运行|7集宏平均|512题总体EM|6集宏平均（敏感性）|HotpotQA|总体EM相对baseline变化|',
 '|---|---:|---:|---:|---:|---:|']
 for x in comparisons:report.append('|'+x['label']+' ('+x['run']+')|'+ '|'.join(f'{x[m]*100:.3f}%' for m in METRICS)+f"|{x['micro512_delta_pp']:+.3f} pp ({x['micro512_relative_change_pct']:+.2f}% relative)|")
 report += ['', '## 原始共同验证点','', '|运行|step|7集宏平均|512题总体EM|正确题数|HotpotQA|Hotpot正确题数|Bamboogle|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for rid,label in PRIMARY.items():
  for s in common:
   x=curves[byid[rid]['local_id']][s]
   report.append(f"|{label}|{s}|{x['macro7']*100:.3f}%|{x['micro512']*100:.3f}%|{x['correct512']}/512|{x['hotpotqa']*100:.3f}%|{x['hotpot_correct']}/73|{int(x['bamboogle'])}/1|")
 report += ['', '## 所有完成验证的0.5B相关运行（不同配置不混合平均）','', '|run|name|λ|α|训练源|评估步|7集宏平均曲线均值|512题总体曲线均值|末次总体EM|最佳总体EM|','|---|---|---:|---:|---|---|---:|---:|---:|---:|']
 for x in summary:
  report.append(f"|{x['run']}|{x['name']}|{x['lambda_distill']}|{x['alpha']}|{x['train_source']}|{x['steps']}|{x['macro7_all_available_mean']*100:.3f}%|{x['micro512_all_available_mean']*100:.3f}%|{x['micro512_last']*100:.3f}%|{x['micro512_best']*100:.3f}%|")
 report += ['', '## 关键解释与限制','',
 '1. Bamboogle只有1/512题，却占7集宏平均的1/7权重。该题从错到对使Avg增加14.286pp，但micro512只增加0.195pp。ER alpha1在step100比baseline的宏平均优势15.813pp中，14.286pp来自该题；两点平均优势6.985pp中，7.143pp来自该题（其他6集总体抵消一部分）。',
 '2. 当前名为alpha2的1k442z39运行，其冻结配置和历史记录实际alpha=1.5，不能当作alpha2实验。它在step100的HotpotQA为30/73，比baseline26/73多4题，但step50为22/73，比baseline25/73少3题；两个点平均只多0.5题，不能据单点判断稳定收益。',
 '3. alpha.5从头运行6bpwblvv/hpwy62o5都只有step50验证；重启3f42x8wd没有验证。gqcv1nnu从旧step50权重继续，不能与新训练seed等同，也不与前一段拼成无缝优化器续训曲线。',
 '4. 更早SOD、DGPO、OPD其他λ运行全部保留表中；它们与主ER的λ、算法、rollout数等不同。only-GRPO的旧运行训练源未筛HotpotQA，模型路径也较旧，不当成严格匹配对照。无GRPO/n=1旧消融单独保留，不用其崩溃证明当前ER有效。',
 '5. 主比较仍有val_batch_size512vs256、teacher_target执行路径、不同日期代码等差异；config_differences.json逐项列出配置差异。不能视作严格单变量消融。',
 '6. 未从历史聚合指标计算置信区间或p值：缺少完整逐题配对预测与预先指定多seed实验。现有多次调参和反复验证也有选择偏差。',
 '7. 本次只读取已有数据并在CPU汇总，没有新增GPU实验、停止训练或改变配置。下一步先修正展示指标，优先检查现有step100/150同题预测；若继续研究，再做固定评估与匹配seed，暂不据这些曲线宣称ER稳定有效。','',
 '训练数据仅HotpotQA与评估包含多个数据集并不矛盾。验证集构成：'+str(COUNTS),
 '复核脚本：scripts/diagnostics/compare_er_baseline_curves.py（searchr1 Python读取W&B）；scripts/diagnostics/summarize_er_baseline_curves.py（CPU汇总）。']
 (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
