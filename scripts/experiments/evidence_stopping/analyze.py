"""Question-paired analysis automatically runs after the full follow-up."""
import collections
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

spec=importlib.util.spec_from_file_location('followup',Path(__file__).with_name('run.py'))
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
OUT=R.OUT


def main():
    p=R.M.read(OUT/'protocol.frozen.json');t=R.rows(OUT/'trajectories.jsonl');f=R.rows(OUT/'forced_answers.jsonl')
    assert R.M.read(OUT/'status.json')['state']=='complete'
    assert len(t)==p['expected_trajectories'] and len(f)==p['expected_forced']
    tx={(r['label'],r['qid'],r['condition']):r for r in t};fx={(r['label'],r['qid']):r for r in f}
    assert len(tx)==len(t) and len(fx)==len(f)
    labels=[c['label'] for c in p['checkpoints']]
    expected={(l,q,c) for l in labels for q in p['selected_qids'] for c in p['conditions']}
    expected|={(l,q,'natural') for l in labels for q in p['natural_qids']}
    assert set(tx)==expected
    assert set(fx)=={(l,q) for l in labels for q in p['selected_qids']}
    for path,h in p['source_hashes'].items():assert R.checksum(Path(path))==h
    assert R.M.digest(R.M.read(OUT/'review.json'))==p['review_hash']
    assert R.M.digest(R.M.read(OUT/'candidates.json'))==p['candidates_hash']
    assert R.M.digest(R.M.read(OUT/'natural_panel.json'))==p['natural_hash']
    for path,saved in p['weights'].items():
        s=Path(path).stat();assert (s.st_size,s.st_mtime_ns)==(saved['size'],saved['mtime_ns'])
    for r in t:
        assert r['search_calls'] <= (4 if r['condition']=='natural' else 3)
        assert not r['final_em'] or r['stop_reason']=='answer_closed'
    checks=dict(complete=True,unique_records=True,coverage=True,source_hashes=True,input_hashes=True,
        weight_metadata_unchanged=True,trajectories=len(t),forced=len(f))
    R.M.save(OUT/'ANALYSIS_CHECKS.json',checks)
    R.summarize(t,f)
    comparisons=[]
    for step_name,steps in [('50',[50]),('100',[100]),('mean50_100',[50,100])]:
        for condition in list(p['conditions'])+['natural']:
            qids=p['natural_qids'] if condition=='natural' else p['selected_qids']
            rng=np.random.default_rng(2026091508)
            ix=rng.integers(0,len(qids),size=(20000,len(qids)))
            metrics=['immediate_em','final_em','search_calls']
            if condition=='both_hops':metrics+=['forced_em','forced_alias_em','forced_minus_immediate_em']
            def get(label,q,metric):
                r=tx[label,q,condition]
                if metric=='forced_minus_immediate_em':return float(fx[label,q]['forced_em'])-float(r['immediate_em'])
                if metric.startswith('forced_'):return float(fx[label,q][metric])
                return float(r[metric])
            for metric in metrics:
                a=np.array([np.mean([get(f'opd-step{s}',q,metric) for s in steps]) for q in qids])
                b=np.array([np.mean([get(f'er-alpha1.5-step{s}',q,metric) for s in steps]) for q in qids])
                d=b-a;lo,hi=np.quantile(d[ix].mean(1),[.025,.975]);scale=1 if metric=='search_calls' else 100
                comparisons.append(dict(step=step_name,condition=condition,metric=metric,n_questions=len(qids),
                    baseline=float(a.mean()),er=float(b.mean()),delta=float(d.mean()*scale),
                    delta_units='calls' if scale==1 else 'percentage_points',
                    relative_change_pct=float((b.mean()/a.mean()-1)*100) if a.mean() else None,
                    paired_bootstrap_low=float(lo*scale),paired_bootstrap_high=float(hi*scale)))
    R.M.save(OUT/'paired_comparisons.json',comparisons)
    with (OUT/'paired_comparisons.csv').open('w') as out:
        w=csv.DictWriter(out,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
    # Audit whether the intervention itself altered already-answering behavior.
    replay=[]
    for label in labels:
        rs=[r for r in f if r['label']==label]
        answered=[r for r in rs if r['ordinary_action']=='answer']
        searched=[r for r in rs if r['ordinary_action']=='search']
        replay.append(dict(label=label,all_cases=len(rs),available=sum(r['available'] for r in rs),
            native_boundaries=sum(r.get('native_reasoning_boundary',False) for r in rs),
            already_answering=len(answered),replay_text_disagreement=sum(r['forced_answer']!=r['ordinary_content'] for r in answered),
            replay_em_disagreement=sum(r['forced_em']!=r['ordinary_immediate_em'] for r in answered),
            originally_searching=len(searched),forced_correct_among_searching=sum(r['forced_em'] for r in searched)))
    R.M.save(OUT/'intervention_checks.json',replay)
    lines=['# Search/answer follow-up: completed paired analysis','',
        f"Complete: {len(t)} trajectories and {len(f)} forced answers. All frozen-source/input/weight metadata checks passed.",
        'The previous 29-question pilot is excluded. Two training steps are clustered within question, never treated as independent seeds.',
        'Question-bootstrap intervals are exploratory and conditional on these historical runs, not training-seed uncertainty; no multiplicity correction.', '',
        '|Step|Condition|Metric|OPD|ER|ER minus OPD|95% paired interval|',
        '|---|---|---|---:|---:|---:|---:|']
    for r in comparisons:
        if r['condition'] not in ('both_hops','natural'):continue
        scale=1 if r['metric']=='search_calls' else 100
        lines.append(f"|{r['step']}|{r['condition']}|{r['metric']}|{r['baseline']*scale:.2f}|{r['er']*scale:.2f}|{r['delta']:+.2f} {r['delta_units']}|[{r['paired_bootstrap_low']:+.2f}, {r['paired_bootstrap_high']:+.2f}]|")
    lines+=['','## Intervention diagnostics','',
        '|Model|Available|Native thought boundary|Already answering|Replay EM changed|Originally searching|Forced correct among searchers|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in replay:lines.append(f"|{r['label']}|{r['available']}/{r['all_cases']}|{r['native_boundaries']}|{r['already_answering']}|{r['replay_em_disagreement']}|{r['originally_searching']}|{r['forced_correct_among_searching']}|")
    lines+=['','## Interpretation constraints','',
        'Consult REPORT.md for every condition and initial checkpoint. Do not infer better stopping merely from lower query counts. Require final accuracy and incomplete-evidence harms together.',
        'A smaller OPD/ER gap under forced answers is consistent with an action-selection contribution, but is not a pure causal decomposition: each model supplied its own preceding reasoning and forcing changes the output prefix.',
        'If replay controls disagree materially, examine token-boundary and BF16 generation effects before treating the forced-answer contrast as diagnostic.',
        'Natural end-to-end metrics use question-only prompts. Controlled post-observation trajectories use supplied gold/non-gold documents and are reported separately.',
        'Reference EM stays primary. Predeclared aliases apply only to the controlled panel; further semantic reviews must be labeled post-hoc and must preserve original metrics.','']
    (OUT/'ANALYSIS.md').write_text('\n'.join(lines))
    print('ANALYSIS_COMPLETE',len(comparisons),flush=True)


if __name__=='__main__':main()
