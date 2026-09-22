"""Audit completed pilot and compute question-paired descriptive comparisons."""
import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location('evidence_use', Path(__file__).with_name('experiment.py'))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
OUT = M.OUT


def write_csv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def value(r, metric):
    if metric == 'answer_rate': return float(r['action'] == 'answer')
    if metric == 'search_rate': return float(r['action'] == 'search')
    if metric == 'invalid_rate': return float(r['action'] == 'invalid')
    if metric == 'usable_next_action': return float(r['retrieval_success'] or r['answer_em'])
    return float(r[metric])


def main():
    protocol = M.read(OUT / 'protocol.frozen.json')
    results = [json.loads(s) for s in (OUT / 'results.jsonl').read_text().splitlines()]
    panel = M.read(OUT / 'panel.json'); review = M.read(OUT / 'review.json')
    qids = protocol['selected_qids']; qmap = {q['qid']: q for q in panel}
    labels = [c['label'] for c in protocol['checkpoints']]
    expected = {(l, q, c) for l in labels for q in qids for c in M.CONDITIONS}
    actual = [(r['label'], r['qid'], r['condition']) for r in results]
    checks = dict(complete=M.read(OUT / 'status.json')['state'] == 'complete',
        no_failure_file=not (OUT / 'failure.json').exists(),
        expected_580=len(expected) == 580,
        exact_coverage=set(actual) == expected and len(actual) == len(expected),
        unique_keys=len(actual) == len(set(actual)),
        panel_hash=M.digest(panel) == protocol['panel_sha256'],
        review_hash=M.digest(review) == protocol['review_sha256'],
        source_hash=hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest() == protocol['script_sha256'])
    weights_unchanged = True
    for path, before in protocol['weight_files'].items():
        s = Path(path).stat()
        weights_unchanged &= s.st_size == before['size'] and s.st_mtime_ns == before['mtime_ns']
    checks['weights_size_mtime_unchanged'] = bool(weights_unchanged)
    for r in results:
        action, content, _ = M.parse_action(r['raw_text'])
        assert (action, content) == (r['action'], r['content'])
        assert M.exact(content if action == 'answer' else None, qmap[r['qid']]['gold']) == r['answer_em']
        assert any(M.norm(t) == M.norm(r['target_title']) for t in r['retrieval_titles']) == r['retrieval_success']
    checks['raw_metrics_recomputed'] = True
    assert all(checks.values()), checks
    M.save(OUT / 'ANALYSIS_CHECKS.json', checks)

    groups = collections.defaultdict(list)
    indexed = {}
    for r in results:
        groups[r['label'], r['condition']].append(r)
        indexed[r['label'], r['qid'], r['condition']] = r
    counts = []
    for (label, condition), rows in groups.items():
        answer = sum(r['action'] == 'answer' for r in rows)
        correct = sum(r['answer_em'] for r in rows)
        counts.append(dict(label=label, condition=condition, n=len(rows),
            searches=sum(r['action'] == 'search' for r in rows), answers=answer,
            invalid=sum(r['action'] == 'invalid' for r in rows),
            gold_document_retrieved=sum(r['retrieval_success'] for r in rows),
            immediate_reference_em=correct, answer_non_em=answer-correct,
            em_among_answers=correct/answer if answer else None,
            average_generated_tokens=float(np.mean([r['new_tokens'] for r in rows]))))
    write_csv(OUT / 'action_counts.csv', counts)

    rng = np.random.default_rng(20260915)
    resamples = rng.integers(0, len(qids), size=(20000, len(qids)))
    comparisons = []
    metrics = ('retrieval_success', 'answer_em', 'answer_rate', 'search_rate', 'usable_next_action')
    for step_name, steps in [('50', [50]), ('100', [100]), ('mean50_100', [50, 100])]:
        for condition in M.CONDITIONS:
            for metric in metrics:
                a = np.array([np.mean([value(indexed[f'opd-step{s}', q, condition], metric) for s in steps]) for q in qids])
                b = np.array([np.mean([value(indexed[f'er-alpha1.5-step{s}', q, condition], metric) for s in steps]) for q in qids])
                delta = b-a
                low, high = np.quantile(delta[resamples].mean(axis=1), [.025, .975])
                comparisons.append(dict(step=step_name, condition=condition, metric=metric, n_questions=len(qids),
                    baseline=float(a.mean()), er=float(b.mean()), delta_pp=float(delta.mean()*100),
                    relative_change_pct=float((b.mean()/a.mean()-1)*100) if a.mean() else None,
                    paired_bootstrap_low_pp=float(low*100), paired_bootstrap_high_pp=float(high*100),
                    er_higher_questions=int((delta>0).sum()), er_lower_questions=int((delta<0).sum()),
                    tied_questions=int((delta==0).sum())))
    write_csv(OUT / 'paired_comparisons.csv', comparisons)
    M.save(OUT / 'paired_comparisons.json', comparisons)

    # Export method-blinded, deduplicated non-EM answers for semantic follow-up.
    # These are NOT silently reclassified or merged into the frozen primary EM.
    cards = {}
    for r in results:
        if r['action'] != 'answer' or r['answer_em']: continue
        key = (r['qid'], r['content'])
        if key not in cards:
            q = qmap[r['qid']]
            cards[key] = dict(question=q['question'], reference=q['gold'], answer=r['content'],
                first_support=q['first_support'], second_support=q['second_support'],
                semantic_label=None, note='Post-hoc review; preserve original reference EM.')
    M.save(OUT / 'semantic_review_cards.json', list(cards.values()))
    lines = ['# GPU 5 evidence-use pilot: audited analysis', '',
        'All 580/580 records are present and unique. Frozen inputs/review/code hashes match;',
        'checkpoint file sizes and modification times are unchanged. Runtime: 732.24 seconds.', '',
        '## Raw action counts', '',
        '|Checkpoint|Condition|N|Search|Answer|Invalid|Gold doc@3|Immediate reference EM|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for c in counts:
        lines.append(f"|{c['label']}|{c['condition']}|{c['n']}|{c['searches']}|{c['answers']}|{c['invalid']}|{c['gold_document_retrieved']}|{c['immediate_reference_em']}|")
    lines += ['', '## Paired checkpoint comparisons', '',
        'ER minus OPD. Bootstrap samples questions (20,000 replicates, seed 20260915).',
        'Shared-step means keep both observations of each question together; steps are not independent seeds.',
        'Intervals are exploratory, conditional on these historical runs and selected panel, unadjusted for multiple comparisons.', '',
        '|Step|Condition|Metric|OPD|ER|Difference (pp)|Question-bootstrap 95% interval (pp)|',
        '|---|---|---|---:|---:|---:|---:|']
    for x in comparisons:
        if (x['condition'] in ('first_hop', 'with_distractors') and x['metric']=='retrieval_success') or (x['condition']=='both_hops' and x['metric'] in ('answer_em','answer_rate')):
            lines.append(f"|{x['step']}|{x['condition']}|{x['metric']}|{x['baseline']:.1%}|{x['er']:.1%}|{x['delta_pp']:+.1f}|[{x['paired_bootstrap_low_pp']:+.1f}, {x['paired_bootstrap_high_pp']:+.1f}]|")
    lines += ['', '## Findings and limits', '',
        '1. The prespecified second-hop retrieval hypothesis is not supported: ER ties OPD at step50 and trails it by 2/29 questions at step100. With distractors ER trails by 4/29 and 2/29.',
        '2. The strongest behavioral difference is answer vs search at sufficient-evidence states. With both gold docs, ER answers on 18/29 and 16/29 questions, vs OPD 4/29 and 7/29. Immediate reference EM is 12/29 and 11/29 vs 4/29 and 5/29.',
        '3. This is not an established increase in reading ability: initial model immediate EM with both docs is 14/29, above both trained ER checkpoints. ER may retain an existing ability to answer rather than acquire a new one.',
        '4. ER also answers early under first-hop plus distractors on 5/29 and 4/29 questions; all nine outputs are reference-EM misses, and inspection identifies wrong entities, years or requested answer types. OPD emits no answers in that condition. More stopping is not uniformly better stopping.',
        '5. EM is not semantic correctness. Examples: Ralph Stanley vs Ralph Edmund Stanley; 50 vs fifty-word; nineteenth century vs nineteenth. Raw EM is unchanged and non-EM answers are exported for separate semantic review.',
        '6. The previous usable-next-action union is not informative about stopping quality under both_hops: it credits retrieving a document already supplied. Do not use that union in this condition to claim task improvement.',
        '7. Two steps from one run per method, 29 structurally selected bridge questions, manually supplied observations, and a neutral forced first query do not establish natural end-to-end benefit or generality. Current overall training-curve evidence still lacks stable improvement.',
        '', '## Concrete paired example (dev_6057, step100)', '',
        'Question: Brad Budde played professionally for a team in the NFL that was founded in 1960 as the Dallas Texans by who?',
        'Both models receive the same two supporting documents. Reference: Lamar Hunt.', '']
    for method in ('opd', 'er-alpha1.5'):
        r = indexed[f'{method}-step100', 'dev_6057', 'both_hops']
        lines += [f'**{method}**', '', '```text', r['text'], '```', '']
    lines += ['## Next discriminating experiment', '',
        'Before a larger training sweep: (a) supply sufficient evidence and compare normal continuation with a forced answer-only continuation, including the initial model; (b) repeat missing-evidence and distractor conditions to measure premature answers; (c) continue natural trajectories to termination and jointly report final accuracy and extra search calls. If forcing answers closes the OPD/ER gap, the main difference is action selection; if a gap remains, answer construction may also differ. Validate on fresh questions after freezing this new hypothesis.', '']
    (OUT / 'ANALYSIS.md').write_text('\n'.join(lines))
    print(json.dumps(checks, indent=2))
    for x in comparisons:
        if x['step']=='mean50_100' and (x['metric']=='retrieval_success' or x['condition']=='both_hops' and x['metric'] in ('answer_em','answer_rate')):
            print(json.dumps(x))


if __name__ == '__main__': main()
