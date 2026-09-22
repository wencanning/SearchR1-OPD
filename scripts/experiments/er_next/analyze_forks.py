"""Analyze COMPLETE fork runs. Replicates are not independent questions."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np


def crossfit_value(manifest, by_key):
    """Fixed 4-fold low-capacity probe. No evaluation rewards fit the predictor.

    Intervals are conditional on these fitted folds; they do not capture
    representation/model selection or independent-training-seed variance.
    """
    ids = manifest['state_ids']; n = manifest['replicates']; split = n // 2
    X, logodds, train_q, eval_q, folds = [], [], [], [], []
    for s in ids:
        pref = by_key[s,'answer',0]['preference']
        l = pref['logp_answer'] - pref['logp_search']
        X.append(pref['hidden_state'] + [l]); logodds.append(l)
        train_q.append([np.mean([by_key[s,b,r]['em'] for r in range(split)]) for b in ('answer','search')])
        eval_q.append([np.mean([by_key[s,b,r]['em'] for r in range(split,n)]) for b in ('answer','search')])
        folds.append(int(hashlib.sha256(by_key[s,'answer',0]['qid'].encode()).hexdigest()[:8],16) % 4)
    X=np.asarray(X); logodds=np.asarray(logodds); train_q=np.asarray(train_q); eval_q=np.asarray(eval_q); folds=np.asarray(folds)
    predicted=np.zeros(len(ids),dtype=int); bias_action=np.zeros(len(ids),dtype=int)
    for fold in range(4):
        tr=folds!=fold; te=~tr
        if tr.sum()<4 or te.sum()<2:
            return dict(status='insufficient_fold_coverage', counts=np.bincount(folds,minlength=4).tolist())
        mean=X[tr].mean(0); scale=X[tr].std(0); scale[scale<1e-6]=1
        x=(X[tr]-mean)/scale; z=(X[te]-mean)/scale
        y=train_q[tr,1]-train_q[tr,0]; intercept=y.mean()
        # Dual ridge with fixed lambda=10; intercept unpenalized.
        coefficient=np.linalg.solve(x@x.T+10*np.eye(tr.sum()), y-intercept)
        pred=z@x.T@coefficient+intercept
        predicted[te]=np.where(pred>0,1,np.where(pred<0,0,(logodds[te]<0).astype(int)))
        boundaries=np.sort(np.unique(-logodds[tr]))
        candidates=np.r_[boundaries[0]-1, (boundaries[:-1]+boundaries[1:])/2, boundaries[-1]+1, 0.]
        # Tie rule: smallest absolute bias, then numeric value. Test rewards unseen.
        bias=max(candidates,key=lambda b:(float(np.mean(train_q[tr,(logodds[tr]+b<0).astype(int)])),-abs(b),-b))
        bias_action[te]=(logodds[te]+bias<0).astype(int)
    value=eval_q[np.arange(len(ids)),predicted]
    controls=dict(fitted_constant_bias=eval_q[np.arange(len(ids)),bias_action],always_answer=eval_q[:,0],always_search=eval_q[:,1])
    rng=np.random.default_rng(2026091613); indices=rng.integers(0,len(ids),size=(20000,len(ids)))
    comparisons={}
    for name,control in controls.items():
        diff=value-control; lo,hi=np.quantile(diff[indices].mean(1),[.05/6,1-.05/6])
        comparisons[name]=dict(delta=float(diff.mean()),ci98_333=[float(lo),float(hi)])
    passed=all(c['ci98_333'][0]>0 for c in comparisons.values())
    return dict(status='descriptive_only' if manifest['exploratory'] else 'confirmation',
        comparisons=comparisons, descriptive_positive_intervals=passed,
        authorizes_training=False,
        qualification='Question bootstrap conditional on fitted folds; not an unconditional learning-algorithm CI. Confirm with a separately locked train/test predictor before a paper claim.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.out / 'execution.frozen.json').read_text())
    rows = [json.loads(l) for l in (args.out / 'branches.jsonl').read_text().splitlines()]
    expected = {(s,b,r) for s in manifest['state_ids'] for b in ('answer','search') for r in range(manifest['replicates'])}
    by_key = {(r['state_id'],r['branch'],r['replicate']): r for r in rows}
    assert len(rows) == len(by_key) and set(by_key) == expected, 'Incomplete or duplicate output'
    assert manifest['replicates'] >= 4, 'At least 4 replicates needed for disjoint selection/evaluation seeds'
    n = manifest['replicates']; split = n // 2
    states = []
    for s in manifest['state_ids']:
        data = {b: [by_key[s,b,r] for r in range(n)] for b in ('answer','search')}
        qs = {r['qid'] for batch in data.values() for r in batch}
        assert len(qs) == 1
        answer = np.mean([r['em'] for r in data['answer']]); search = np.mean([r['em'] for r in data['search']])
        delta_train = np.mean([r['em'] for r in data['search'][:split]]) - np.mean([r['em'] for r in data['answer'][:split]])
        # Ties keep reference canonical argmax; never use held-out returns to select.
        reference = 'answer' if data['answer'][0]['preference']['answer_probability'] >= .5 else 'search'
        selected = 'search' if delta_train > 0 else 'answer' if delta_train < 0 else reference
        eval_value = {b: float(np.mean([r['em'] for r in data[b][split:]])) for b in data}
        states.append(dict(state_id=s, qid=next(iter(qs)), answer_em=float(answer), search_em=float(search),
            delta=float(search-answer), selected_from_first_seeds=selected, reference_action=reference,
            heldout_selected_em=eval_value[selected], heldout_reference_em=eval_value[reference],
            heldout_answer_em=eval_value['answer'], heldout_search_em=eval_value['search'],
            mean_search_calls=float(np.mean([r['search_calls'] for r in data['search']])),
            both_branches_never_correct=bool(answer==0 and search==0)))
    assert len({s['qid'] for s in states}) == len(states), 'Use question-cluster bootstrap for multiple states per question'
    diff = np.array([s['heldout_selected_em'] - s['heldout_reference_em'] for s in states])
    rng = np.random.default_rng(2026091611)
    lo, hi = np.quantile(diff[rng.integers(0,len(diff),size=(20000,len(diff)))].mean(1), [.025,.975])
    result = dict(n_questions=len(states), n_branches=len(rows), states=states,
        heldout_gain_over_reference=float(diff.mean()), ci95=[float(lo),float(hi)],
        stop_reasons=dict(collections.Counter(r['stop_reason'] for r in rows)),
        observed_both_never_correct=sum(s['both_branches_never_correct'] for s in states),
        qualification='Exploratory; finite samples do not establish latent knowledge sufficiency or trainability. No learned policy tested.')
    result['crossfit_probe']=crossfit_value(manifest,by_key)
    oracle_comparisons={}
    indices=rng.integers(0,len(states),size=(20000,len(states)))
    for action in ('answer','search'):
        d=np.array([s['heldout_selected_em']-s['heldout_'+action+'_em'] for s in states])
        lower,upper=np.quantile(d[indices].mean(1),[.0125,.9875])
        oracle_comparisons['always_'+action]=dict(delta=float(d.mean()),ci97_5=[float(lower),float(upper)])
    result['sample_split_oracle_diagnostic']=oracle_comparisons
    (args.out / 'fork_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    report=[
        '# 同状态回答/搜索分支诊断（探索性）','',
        f"{len(states)}个不同问题；{len(rows)}个分支续写；未完成续写全部保留。",
        f"所有采样中两支都未答对：{result['observed_both_never_correct']}/{len(states)}（不等同于永远不可解）。",
        f"前半seed选动作、后半seed评估，相对规范动作argmax的收益差：{100*diff.mean():+.2f}pp，95%题目区间[{100*lo:+.2f},{100*hi:+.2f}]。",
        '该选择器使用当前题目的分支结果，是信息价值诊断；不是可部署策略，也没有证明从上下文可学习。',
        '下一阶段必须用训练题学习动作预测器，在全新问题和自由生成中检验。',
        '只估计固定token/检索预算下的干预收益；无答案不自动意味着应该检索。','',
        '|独立seed选择器对照|收益差(pp)|97.5%题目区间(pp)|', '|---|---:|---|']
    for name,value in oracle_comparisons.items():
        report.append(f"|{name}|{100*value['delta']:+.2f}|[{100*value['ci97_5'][0]:+.2f},{100*value['ci97_5'][1]:+.2f}]|")
    report += ['', 'cross-fit预测器比较见fork_summary.json；所有区间只作当前探索性描述，脚本不会授权训练。']
    (args.out / 'FORK_REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='states'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
