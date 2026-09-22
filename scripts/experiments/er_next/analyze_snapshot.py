"""Analyze a frozen prefix of fork outputs; incomplete questions stay excluded."""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'


def main():
    import numpy as np
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rows', type=int, required=True)
    args = p.parse_args()
    manifest = json.loads((args.out/'execution.frozen.json').read_text())
    lines = (args.out/'branches.jsonl').read_text().splitlines()[:args.rows]
    assert len(lines) == args.rows
    raw = '\n'.join(lines)+'\n'
    snapshot = args.out/f'snapshot_{args.rows}'
    snapshot.mkdir(exist_ok=True)
    frozen = snapshot/'branches.frozen.jsonl'
    if frozen.exists():
        assert frozen.read_text() == raw
    else:
        frozen.write_text(raw)
    rows = [json.loads(l) for l in lines]
    keys = {(r['state_id'], r['branch'], r['replicate']) for r in rows}
    n = manifest['replicates']
    expected = {(s,b,i) for s in manifest['state_ids'] for b in ('answer','search') for i in range(n)}
    assert len(keys) == len(rows) and keys <= expected
    states_path = Path('reports/er_next_20260916/fork_states.frozen.json')
    assert hashlib.sha256(states_path.read_bytes()).hexdigest() == manifest['states_sha256']
    states = {s['state_id']:s for s in json.loads(states_path.read_text())['states']}
    helper = Path('scripts/experiments/evidence_use/experiment.py')
    assert hashlib.sha256(helper.read_bytes()).hexdigest() == manifest['helper_sha256']
    spec = importlib.util.spec_from_file_location('existing_score', helper)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert all(module.exact(r['final_answer'], states[r['state_id']]['gold']) == r['em'] for r in rows)
    by = collections.defaultdict(list)
    for r in rows:
        by[r['state_id']].append(r)
    complete = {s:rs for s,rs in by.items() if len(rs)==2*n}
    assert len({states[s]['qid'] for s in complete}) == len(complete)
    flat = [r for rs in complete.values() for r in rs]
    tok = AutoTokenizer.from_pretrained(manifest['checkpoint'], local_files_only=True)
    table = []
    evidence = collections.Counter()
    for sid, rs in complete.items():
        actions = {b: sorted([r for r in rs if r['branch']==b],key=lambda r:r['replicate']) for b in ('answer','search')}
        ac = sum(r['em'] for r in actions['answer']); sc = sum(r['em'] for r in actions['search'])
        half = n//2
        deltas = [sum(r['em'] for r in actions['search'][slice_])-sum(r['em'] for r in actions['answer'][slice_]) for slice_ in (slice(0,half),slice(half,n))]
        table.append(dict(qid=states[sid]['qid'],answer_correct=ac,search_correct=sc,replicates=n,
                          delta=(sc-ac)/n,half_seed_deltas=deltas,
                          new_search_calls=sum(r['search_calls'] for r in actions['search']),
                          answer_probability=actions['answer'][0]['preference']['answer_probability']))
        for r in actions['search']:
            visible = [tok.decode(a['observation_ids'],skip_special_tokens=False) for a in r['actions'] if 'observation_ids' in a]
            hit = any(module.mention(g,v) for g in states[sid]['gold'] for v in visible)
            evidence[f"{'hit' if hit else 'no_hit'}_{'correct' if r['em'] else 'wrong'}"] += 1
    delta = np.array([r['delta'] for r in table])
    rng = np.random.default_rng(2026091609)
    ci = np.quantile(delta[rng.integers(0,len(delta),size=(20000,len(delta)))].mean(1),[.025,.975]).tolist()
    arms = {}
    for b in ('answer','search'):
        arm = [r for r in flat if r['branch']==b]
        arms[b] = dict(n=len(arm),correct=sum(r['em'] for r in arm),em=sum(r['em'] for r in arm)/len(arm),
                       stop_reasons=dict(collections.Counter(r['stop_reason'] for r in arm)),
                       new_search_calls=sum(r['search_calls'] for r in arm))
    result = dict(snapshot_rows=len(rows),planned_rows=len(expected),complete_questions=len(complete),
                  incomplete_questions={states[s]['qid']:len(rs) for s,rs in by.items() if s not in complete},
                  unseen_questions=len(manifest['state_ids'])-len(by),arms=arms,paired_delta=float(delta.mean()),
                  question_bootstrap_ci95=ci,search_better=sum(r['delta']>0 for r in table),
                  answer_better=sum(r['delta']<0 for r in table),ties=sum(r['delta']==0 for r in table),
                  both_never_correct=sum(r['answer_correct']==r['search_correct']==0 for r in table),
                  strict_direction_agrees_between_halves=sum(r['half_seed_deltas'][0]*r['half_seed_deltas'][1]>0 for r in table),
                  strict_direction_reverses_between_halves=sum(r['half_seed_deltas'][0]*r['half_seed_deltas'][1]<0 for r in table),
                  visible_gold_occurrence=dict(evidence),sha256=hashlib.sha256(raw.encode()).hexdigest(),
                  qualification='Exploratory observed complete questions; seeds are not independent questions. Gold string occurrence is not evidence sufficiency. No learned policy or ER comparison.')
    (snapshot/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (snapshot/'per_question.json').write_text(json.dumps(table,indent=2)+'\n')
    report = ['# 完整配对问题的阶段性结果','',f"冻结前{len(rows)}个分支，只比较{len(complete)}道已完成两动作全部重复的问题。",'',
              '|动作|答对/总数|EM|新增搜索调用|','|---|---:|---:|---:|']
    for b,r in arms.items():
        report.append(f"|{b}|{r['correct']}/{r['n']}|{100*r['em']:.2f}%|{r['new_search_calls']}|")
    report += ['',f"搜索减直接回答：{100*delta.mean():+.2f}pp；按问题bootstrap的95%描述性区间[{100*ci[0]:+.2f},{100*ci[1]:+.2f}]pp。",
               f"搜索观察正确率较高{result['search_better']}题，直接回答较高{result['answer_better']}题，平局{result['ties']}题；两支均无命中{result['both_never_correct']}题。",
               '未完成/未运行问题不能计为错误，也没有参与本表。当前不是完整64题结论。' if len(rows)<len(expected) else '全部计划分支已完成。',
               '该实验衡量固定学生状态的强制动作续写，不能代表训练曲线准确率或ER相对OPD效果。两次seed半区仅作稳定性诊断，不是独立题目或泛化确认。', '',
               '可见gold完整字符串命中计数见summary.json；含字符串不保证证据充分，未含字符串也不证明无法推理得出答案。', '',
               '|问题|直接回答正确数|搜索正确数|','|---|---:|---:|']
    report += [f"|{r['qid']}|{r['answer_correct']}/{n}|{r['search_correct']}/{n}|" for r in table]
    (snapshot/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
