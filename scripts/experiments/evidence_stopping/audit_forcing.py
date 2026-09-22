"""Post-hoc audit of intervention availability; preserve all primary metrics."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/evidence_stopping_gpu5_20260915'


def load_rows(name):
    return [json.loads(s) for s in (OUT/name).read_text().splitlines()]


def main():
    forced=load_rows('forced_answers.jsonl');traj=load_rows('trajectories.jsonl')
    fx={(r['label'],r['qid']):r for r in forced}
    rng=np.random.default_rng(2026091509);matched=[]
    for step in (50,100):
        a={q:r for (l,q),r in fx.items() if l==f'opd-step{step}'}
        b={q:r for (l,q),r in fx.items() if l==f'er-alpha1.5-step{step}'}
        keys=sorted(q for q in a if a[q]['available'] and b[q]['available'])
        for metric in ('ordinary_immediate_em','forced_em','forced_alias_em'):
            av=np.array([a[q][metric] for q in keys],dtype=float)
            bv=np.array([b[q][metric] for q in keys],dtype=float)
            ix=rng.integers(0,len(keys),size=(20000,len(keys)))
            ci=np.quantile((bv-av)[ix].mean(1),[.025,.975])*100
            matched.append(dict(step=step,metric=metric,n=len(keys),qids=keys,
                baseline_correct=int(av.sum()),er_correct=int(bv.sum()),
                baseline=float(av.mean()),er=float(bv.mean()),
                delta_pp=float((bv-av).mean()*100),ci_low_pp=float(ci[0]),ci_high_pp=float(ci[1]),
                er_only_correct=int(((bv==1)&(av==0)).sum()),opd_only_correct=int(((av==1)&(bv==0)).sum())))
    replay=[]
    for label in sorted({r['label'] for r in forced}):
        rs=[r for r in forced if r['label']==label]
        tested=[r for r in rs if r['available'] and r['ordinary_action']=='answer']
        replay.append(dict(label=label,total=len(rs),unavailable=sum(not r['available'] for r in rs),
            unavailable_but_ordinary_correct=sum(not r['available'] and r['ordinary_immediate_em'] for r in rs),
            actual_answer_replays=len(tested),actual_replay_em_changed=sum(r['forced_em']!=r['ordinary_immediate_em'] for r in tested)))
    result=dict(post_hoc=True,interpretation='Common-availability sensitivity analysis, not randomized missingness or a full-population causal comparison.',
                matched=matched,replay=replay)
    (OUT/'MATCHED_FORCING_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 完成后的结果解读与强制回答口径复核','',
        '1,000 条轨迹与 170 条强制回答记录全部完成，约 58 分钟。170 条是计划记录数，其中 11 条未执行强制回答，按预设规则计为失败；159 条实际执行了干预。原始结果及预设指标不修改。','',
        '## 强制回答的关键口径','',
        'ER 在 step50 有 4/34 条、step100 有 7/34 条直接输出搜索或答案而没有推理段，因此未满足“保留完整推理后强制回答”的预设门槛。OPD 34/34 均满足。全样本 forced EM 把不可执行干预记为 0，是操作性指标，不能直接解释成阅读/作答能力差距。',
        '特别是 ER step100 两条本来已经答对的输出（45%、Via Vai）也因没有推理段而被记为 forced EM=0。这不是实际强制生成后答错。','',
        '下面在每个训练步上，用双方均可执行强制回答的相同题目比较。该分析为事后共同子集敏感性分析，不能消除依赖模型行为的样本选择偏差。','',
        '|训练步|共同题数|指标|OPD 正确数|ER 正确数|ER−OPD (pp)|95%题目配对区间|',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in matched:
        lines.append(f"|{r['step']}|{r['n']}|{r['metric']}|{r['baseline_correct']}|{r['er_correct']}|{r['delta_pp']:+.1f}|[{r['ci_low_pp']:+.1f}, {r['ci_high_pp']:+.1f}]|")
    lines+=['','两个训练步的共同子集上，强制回答的原始 EM 和预声明别名 EM 均持平。这与差异主要涉及行动选择的解释相容，但不证明两个模型整体能力相同。',
        '实际已作答样本的重放 EM 变化：OPD step50 为 1 例，ER step50 为 1 例，其余为 0。此前自动表格 ER step100 的 2 例“重放变化”实际上是没有执行干预，不能计为真实重放失败。','',
        '## 完整任务结果','',
        '|条件|模型|步数|题数|即时答对|最终答对|最终别名答对|实际搜索调用总数|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for step in (50,100):
        for condition in ('both_hops','natural','with_distractors'):
            for method in ('opd','er-alpha1.5'):
                rs=[r for r in traj if r['label']==f'{method}-step{step}' and r['condition']==condition]
                lines.append(f"|{condition}|{method}|{step}|{len(rs)}|{sum(r['immediate_em'] for r in rs)}|{sum(r['final_em'] for r in rs)}|{sum(r['final_alias_em'] for r in rs)}|{sum(r['search_calls'] for r in rs)}|")
    lines+=['',
        '64 道自然搜索题：共同两步平均 EM 为 OPD 28.91%、ER 30.47%，差 +1.56 pp；题目配对区间 [-5.47,+8.59] pp。平均搜索调用 1.977→1.805，减少 8.70%；该结果不含训练随机种子不确定性，也不代表已验证总体准确率不下降。',
        '34 道充分证据题：ER 即时 EM 高，但继续搜索至结束后，step50 最终 EM 为 OPD 21/34 vs ER 20/34，step100 为 24/34 vs 19/34。预声明别名计分分别为 28/34 vs 28/34、29/34 vs 25/34。不能把即时 EM 优势写成最终任务收益。',
        '第一跳加干扰：ER 在 step50/100 分别立即作答 6/34、7/34，均未命中参考 EM；OPD 为 0/34、1/34，亦未命中。需进一步语义复核，不能只以“少搜索”判断决策更好。','',
        '当前可支持：这两个历史训练运行在固定证据状态下表现出不同的搜索/作答倾向；ER 更早回答，自然面板平均搜索数更少。尚不支持稳定准确率提升、更强阅读能力，或更准确的停止决策。','']
    (OUT/'INTERPRETATION.md').write_text('\n'.join(lines))
    note='> 后续口径复核：11 条强制回答记录因缺少推理段而未执行干预，不能将其计零直接解释为能力较差。双方均可执行干预的共同子集上，两步 forced EM 均持平。请先看 [INTERPRETATION.md](INTERPRETATION.md)。\n\n'
    for name in ('REPORT.md','ANALYSIS.md'):
        path=OUT/name;text=path.read_text()
        if not text.startswith('> 后续口径复核：'):path.write_text(note+text)
    print(json.dumps(dict(matched=matched,replay=replay),indent=2))


if __name__=='__main__':main()
