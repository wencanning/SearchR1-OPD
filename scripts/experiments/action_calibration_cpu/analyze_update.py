"""Report local changes and token roles without claiming task accuracy gains."""
import collections
import numpy as np
from common import OUT,rows,save,M


def main():
    results=rows(OUT/'update_effects.jsonl');assert len(results)==4
    assert {r['variant'] for r in results}=={'opd','full_er','action_only_er','body_only_er'}
    assert all(r['device']=='cpu' and not r['cuda_initialized'] for r in results)
    effects={}
    for r in results:
        before={(x['qid'],x['condition']):x for x in r['before']}
        after={(x['qid'],x['condition']):x for x in r['after']};assert len(before)==len(after)==24
        effects[r['variant']]={key:dict(log_odds=after[key]['log_odds']-before[key]['log_odds'],
            reference_body_mean_logp=after[key].get('reference_body_mean_logp',0)-before[key].get('reference_body_mean_logp',0)) for key in before}
    qids=sorted({q for q,c in effects['opd']});rng=np.random.default_rng(2026091604);boot=rng.integers(0,8,size=(20000,8))
    summary=[]
    for variant in effects:
        for condition in ['first_hop','both_hops','with_distractors']:
            values=np.array([effects[variant][q,condition]['log_odds'] for q in qids])
            incremental=np.array([effects[variant][q,condition]['log_odds']-effects['opd'][q,condition]['log_odds'] for q in qids])
            low,high=np.quantile(incremental[boot].mean(1),[.025,.975])
            summary.append(dict(variant=variant,condition=condition,n=8,mean_log_odds_update=float(values.mean()),
                mean_increment_over_opd=float(incremental.mean()),ci_low=float(low),ci_high=float(high)))
    save(OUT/'update_summary.json',summary)
    tokens=M.read(OUT/'update_token_signals.json');role_stats=[]
    for role in ['action','action_mixed','format','body']:
        subset=[t for t in tokens if t['role']==role]
        if not subset:continue
        role_stats.append(dict(role=role,n=len(subset),mean_opd_advantage=float(np.mean([t['opd_advantage'] for t in subset])),
            mean_er_increment=float(np.mean([t['er_increment'] for t in subset])),
            sum_abs_er_increment=float(sum(abs(t['er_increment']) for t in subset)),
            advantage_sign_flips=sum(t['opd_advantage']*t['full_er_advantage']<0 for t in subset)))
    save(OUT/'update_role_summary.json',role_stats)
    samples=rows(OUT/'update_samples.jsonl')
    lines=['# ER增量作用位置：共同小批次局部更新','',
        f"8道更新题×2证据条件×2次采样，共32条短续写；其中完整动作{sum(c['complete_action'] for c in samples)}/32。截断或格式异常保留，不赋予虚构失败奖励。",'',
        '四组从同一OPD step50参数和fresh AdamW状态开始，各更新一次。λ=.01，所有组GRPO优势=0，隔离蒸馏分量，不复现完整ER+GRPO训练。所有token保留普通OPD，只切换ER增量位置。','',
        '动作词与混合动作token单列；format是纯协议格式，body包括搜索查询、答案或续写正文。共享参数使未直接使用ER的位置也可能发生概率变化。', '',
        '|变体|探针证据|更新后回答/搜索log-odds变化|相对OPD的额外变化|95%题目区间|', '|---|---|---:|---:|---|']
    for s in summary:lines.append(f"|{s['variant']}|{s['condition']}|{s['mean_log_odds_update']:+.4f}|{s['mean_increment_over_opd']:+.4f}|[{s['ci_low']:+.4f},{s['ci_high']:+.4f}]|")
    lines += ['', '|位置|token数量|ER额外优势均值|ER额外优势绝对值之和|优势翻转数|','|---|---:|---:|---:|---:|']
    for s in role_stats:lines.append(f"|{s['role']}|{s['n']}|{s['mean_er_increment']:+.5f}|{s['sum_abs_er_increment']:.5f}|{s['advantage_sign_flips']}|")
    lines += ['', '|变体|充分证据探针参考答案正文平均logp变化|','|---|---:|']
    for variant in effects:
        body=np.mean([effects[variant][q,'both_hops']['reference_body_mean_logp'] for q in qids])
        lines.append(f'|{variant}|{body:+.5f}|')
    lines += ['', '8道探针取自测试集，和更新8题完全不重叠；单步结果不包含训练seed方差，不证明总体准确率提高或长期行为归因。',
        '参考正文概率是固定参考串的teacher-forcing分数，不是生成EM；规范开标签概率也不覆盖全部行动实现。',
        '若全ER变化能被动作ER复现而正文ER不能，可继续检验动作分量；反之应保留正文/格式/共享参数解释。不得因某组局部概率变化大就宣称主方法有效。',
        '四组使用相同总token损失分母、同学习率与梯度裁剪阈值，但Adam和裁剪是非线性的，分量更新不应被假定可线性相加。','']
    (OUT/'UPDATE_REPORT.md').write_text('\n'.join(lines))
    save(OUT/'status.json',dict(state='complete',bias_calibration_questions=16,bias_test_questions=32,update_questions=8,
        sampled_responses=32,local_update_variants=4,cuda_initialized=False))
    print('ACTION_CALIBRATION_PIPELINE_COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:save(OUT/'failure.json',dict(stage='analyze_update',error=repr(e)));raise
