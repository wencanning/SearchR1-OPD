"""Read cached scores and trajectories; no model loading or GPU calls."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path('reports/evidence_update_cpu_20260914')
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = Path('reports/evidence_utility/delta_table_triq_corrected.parquet')
d = pd.read_parquet(SOURCE).sort_values(['trajectory_id', 'k'])
audit = {'source': str(SOURCE), 'rows': len(d),
         'trajectories': d.trajectory_id.nunique(), 'groups': d.group_id.nunique(),
         'missing_correctness_rows': int(d.answer_correct.isna().sum()),
         'duplicate_step_keys': int(d.duplicated(['trajectory_id', 'k']).sum()),
         'delta_max_error': float((d.delta_k - (d.h_bar_k-d.h_bar_prev)).abs().max()),
         'inconsistent_trajectory_labels': int((d.groupby('trajectory_id').answer_correct.nunique()>1).sum())}
previous = d.groupby('trajectory_id').h_bar_k.shift()
audit['consecutive_score_max_error'] = float((d.h_bar_prev-previous).abs().max())
d['ig'] = -d.delta_k
d['wrong'] = d.answer_correct.eq(0)
valid = d[d.answer_correct.notna()].copy()

def describe(frame, threshold=0):
    positive = frame.ig.gt(threshold)
    wrong = frame.wrong
    both = positive & wrong
    def ratio(n, den):
        return float(n/den) if den else None
    return {'n': len(frame), 'positive_n': int(positive.sum()),
            'wrong_n': int(wrong.sum()), 'positive_wrong_n': int(both.sum()),
            'p_wrong_given_positive': ratio(both.sum(), positive.sum()),
            'p_positive_given_wrong': ratio(both.sum(), wrong.sum()),
            'joint_rate': ratio(both.sum(), len(frame))}

rows = []
for column in ['all', 'data_source', 'global_step', 'k']:
    groups = [('all', valid)] if column == 'all' else valid.groupby(column)
    for name, frame in groups:
        rows.append({'stratum': column, 'value': name, **describe(frame)})
pd.DataFrame(rows).to_csv(OUT/'step_statistics.csv', index=False)
sensitivity = [{'threshold_nats_per_gold_token': x, **describe(valid,x)} for x in [0,.1,.25,.5,1.]]
pd.DataFrame(sensitivity).to_csv(OUT/'threshold_sensitivity.csv',index=False)

# A trajectory is the unit here: do not count every search as an independent answer.
t = d.groupby('trajectory_id', sort=False).agg(
    group_id=('group_id','first'), data_source=('data_source','first'),
    global_step=('global_step','first'), gold=('gold','first'),
    answer_correct=('answer_correct','first'), n_search=('k','size'),
    initial_h=('h_bar_prev','first'), final_h=('h_bar_k','last'),
    total_ig=('ig','sum'), max_ig=('ig','max'), any_hit=('retrieval_hit','max'))
t = t[t.answer_correct.notna()].copy()
t['ig'] = t.total_ig
t['wrong'] = t.answer_correct.eq(0)
t.to_csv(OUT/'trajectory_statistics.csv')
traj = {'net_gain': describe(t), 'any_gain': describe(t.assign(ig=t.max_ig)),
        'by_source': {k:describe(v) for k,v in t.groupby('data_source')}}
audit['missing_correctness_trajectories'] = int(d.trajectory_id.nunique()-len(t))

# Bootstrap (step, question) groups, not correlated tokens or searches.
def cluster_ci(frame):
    a=frame.assign(positive=frame.ig.gt(0), both=frame.ig.gt(0)&frame.wrong)
    grouped=a.groupby('group_id')[['positive','wrong','both']].sum().to_numpy(dtype=float)
    rng=np.random.default_rng(20260914)
    draws=grouped[rng.integers(0,len(grouped),size=(2000,len(grouped)))].sum(axis=1)
    return {name:np.quantile(draws[:,2]/draws[:,col],[.025,.975]).tolist()
            for name,col in [('p_wrong_given_positive',0),('p_positive_given_wrong',1)]}
audit['step_cluster_bootstrap_95ci'] = cluster_ci(valid)
audit['trajectory_cluster_bootstrap_95ci'] = cluster_ci(t)

# Counts within groups containing both correct and wrong trajectories.
mixed_ids=t.groupby('group_id').answer_correct.nunique()
mixed=t[t.group_id.isin(mixed_ids[mixed_ids>1].index)]
differences=[]
for _, frame in mixed.groupby('group_id'):
    differences.append(float(frame.loc[frame.wrong,'total_ig'].mean()-frame.loc[~frame.wrong,'total_ig'].mean()))
audit['mixed_group_count']=len(differences)
audit['mixed_group_wrong_minus_correct_mean_net_ig']=float(np.mean(differences))

# Stream all raw teacher trajectories. Keep an unfiltered, deterministic first-12
# hit/wrong sample for human review, not a best-looking-case sample.
counts=[]
cases=[]
for filename in ['data/teacher_rollout_7b_hotpotqa_train/raw_rollouts.jsonl',
                 'data/teacher_rollout/raw_rollouts.jsonl']:
    c={'file':filename,'n':0,'correct':0,'hit':0,'hit_wrong':0}
    with open(filename) as f:
        for line in f:
            r=json.loads(line)
            c['n']+=1
            hit=bool(r.get('retrieval_hit'))
            wrong=r.get('answer_correct') is False
            c['correct']+=int(r.get('answer_correct') is True)
            c['hit']+=int(hit)
            c['hit_wrong']+=int(hit and wrong)
            if '7b_hotpotqa' in filename and hit and wrong and len(cases)<12:
                cases.append({k:r.get(k) for k in ['sample_id','question','ground_truth','final_answer','trajectory_text','stop_reason','search_turn_count']})
    counts.append(c)
pd.DataFrame(counts).to_csv(OUT/'teacher_rollout_counts.csv',index=False)
with open(OUT/'teacher_cases_first12.json','w') as f:
    json.dump(cases,f,ensure_ascii=False,indent=2)
with open(OUT/'summary.json','w') as f:
    json.dump({'audit':audit,'steps':describe(valid),'trajectories':traj,'teacher_counts':counts},f,ensure_ascii=False,indent=2)
print(json.dumps({'audit':audit,'steps':describe(valid),'trajectories':traj,'teacher_counts':counts},ensure_ascii=False,indent=2))
