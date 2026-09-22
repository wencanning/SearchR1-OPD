"""Read-only attribution audit: strict body here means query body, not answer body."""
import collections
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'reports/action_calibration_cpu_20260916'
OUT=ROOT/'reports/er_generation_cpu_20260916'


def main():
    sources=[OLD/'update_samples.jsonl',OLD/'update_targets.jsonl',ROOT/'reports/strict_role_cpu_20260916/update_token_signals.json']
    samples=[json.loads(l) for l in sources[0].read_text().splitlines()]
    targets={(r['qid'],r['condition'],r['sample']):r for r in [json.loads(l) for l in sources[1].read_text().splitlines()]}
    tokens=json.loads(sources[2].read_text())
    roles={(r['qid'],r['condition'],r['sample'],r['token_index']):r['role'] for r in tokens}
    groups=collections.defaultdict(lambda:collections.Counter());actions=collections.Counter();details=[]
    for row in samples:
        key=row['qid'],row['condition'],row['sample'];target=targets[key]
        match=re.fullmatch(r'\s*<(search|answer)>(.*?)</\1>\s*',row['raw_text'],re.S)
        action=match[1] if match else 'unparsed';actions[action]+=1
        for i,(obs,hid,er) in enumerate(zip(target['observed_log_probs'],target['hidden_log_probs'],target['er_log_probs'])):
            role=roles[*key,i];ratio=obs-hid;increment=er-obs;constant=1.5*ratio-increment
            g=groups[action,role];g['n']+=1
            g['positive_observed_vs_hidden']+=ratio>1e-5
            g['positive_er_vs_observed']+=increment>1e-5
            g['supported_but_reduced']+=ratio>1e-5 and increment< -1e-5
            g['sum_er_increment']+=increment;g['sum_normalizer']+=constant
            details.append(dict(qid=key[0],condition=key[1],sample=key[2],index=i,role=role,action=action,
                observed_minus_hidden=ratio,er_minus_observed=increment,inferred_log_normalizer=constant))
    results=[dict(action=a,role=r,**dict(v)) for (a,r),v in groups.items()]
    output=dict(actions=dict(actions),groups=results,details=details,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/'query_signal_audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    report=['# 严格正文的真实类型与ER信号审计','',
        '28条严格合法续写全部为search；4条无法严格解析。292个严格正文token全部是查询正文，答案正文token=0。',
        '这推翻了“本实验直接测试了答案正文ER学习”的解释。答案参考串logp的改善只能是从查询更新产生的间接影响。', '',
        '|输出类型|位置|token数|证据可见后概率增大|ER比普通教师概率增大|证据支持但ER概率下降|',
        '|---|---|---:|---:|---:|---:|']
    for r in results:report.append(f"|{r['action']}|{r['role']}|{r['n']}|{r['positive_observed_vs_hidden']}|{r['positive_er_vs_observed']}|{r['supported_but_reduced']}|")
    report += ['',
        '查询正文292token中，仅33个ER概率相对普通教师增大；125个在证据可见时比隐藏时更受支持，其中89个经过ER构造后反而低于普通教师概率。阈值1e-5过滤浮点近零。',
        '这些是8个更新问题上的相关token描述，不能当292个独立样本做显著性检验；也未证明这些查询是否有用。', '',
        '## 归一化的精确分解',
        '令r(v)=log p_obs(v)-log p_hidden(v)，p_ER(v)∝p_obs(v)exp(αr(v))。则log p_ER(v)-log p_obs(v)=αr(v)-C(s)，其中C(s)=log Σ_v p_obs(v)exp(αr(v))。',
        'C(s)对同一token前缀的所有候选token相同，可以由现存selected log-prob恢复。这说明“ER增强证据支持token”的直觉不能直接等同于采样token概率都上升。',
        '这不自动构成方法缺陷：固定状态、精确on-policy、无裁剪的score-function梯度中，Eπ[C(s)∇logπ]=0，状态常数是baseline。有限采样、PPO裁剪和多轮更新可能改变方差/行为，必须另行验证，不能用本审计宣布归一化有害。', '',
        '## 研究决定',
        '先完成正在运行的生成迁移pilot，检验查询正文更新有无可见答案收益。若无收益，停止扩充本批次token角色诊断。',
        '若有线索，正式训练必须分开记录query/reasoning/answer覆盖率，并以普通OPD及同预算对照检验。温度锐化或常数baseline解释只能作为待排除的解释，不能预先写成新方法机制。']
    (OUT/'QUERY_SIGNAL_AUDIT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(actions=dict(actions),strict_body_tokens=292,strict_answer_tokens=0),ensure_ascii=False))


if __name__=='__main__':main()
