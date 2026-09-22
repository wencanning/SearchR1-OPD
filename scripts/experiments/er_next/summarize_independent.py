"""CPU-only completeness and visible-evidence audit of a finished fork run.

Lexical gold occurrence is descriptive: it is neither necessary nor sufficient
for answerability. This script never changes the frozen EM labels.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import unicodedata

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ' '.join(''.join(c if c.isalnum() else ' ' for c in text).split())


def contains(text, gold):
    haystack = ' ' + normalize(text) + ' '
    return any(normalize(g) and ' ' + normalize(g) + ' ' in haystack for g in gold)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--states', type=Path, default=Path('reports/er_next_20260916/fork_states.frozen.json'))
    args = p.parse_args()
    out = args.out
    manifest = read(out / 'execution.frozen.json')
    status = read(out / 'status.json')
    rows = [json.loads(l) for l in (out / 'branches.jsonl').read_text().splitlines()]
    expected = {(s, b, r) for s in manifest['state_ids'] for b in ('answer', 'search') for r in range(manifest['replicates'])}
    observed = {(r['state_id'], r['branch'], r['replicate']) for r in rows}
    assert status['state'] == 'complete' and observed == expected and len(rows) == len(expected)
    assert sha(args.states) == manifest['states_sha256']
    code = Path(__file__).with_name('collect_forks.py')
    assert sha(code) == manifest['code_sha256'], 'Collector changed after the run'
    assert sha(code.with_name('run_gpu5_guarded.py')) == manifest['gpu_guard_sha256']
    assert sha(out / 'retriever.frozen.json') == manifest['retriever_manifest_sha256']
    for path, metadata in manifest['weights'].items():
        stat = Path(path).stat()
        assert metadata == dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    states = {s['state_id']: s for s in read(args.states)['states']}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(manifest['checkpoint'], local_files_only=True)
    cache = read(out / 'retrieval_cache.json')
    evidence_rows = []
    examples = ['# Actual continuations and visible evidence', '',
                'Exploratory audit. Gold string occurrence does not establish supporting evidence or answerability.', '']
    for row in rows:
        state = states[row['state_id']]
        prefix = tok.decode(state['prefix_ids'], skip_special_tokens=False)
        original_observations = re.findall(r'<information>(.*?)</information>', prefix, re.S)
        observations = []
        for action in row['actions']:
            if 'observation_ids' not in action:
                continue
            visible = tok.decode(action['observation_ids'], skip_special_tokens=False)
            docs = cache[action['query']]
            raw = '\n'.join(d['document']['contents'] for d in docs)
            observations.append(dict(query=action['query'], visible_text=visible,
                raw_gold_hit=contains(raw, state['gold']), visible_gold_hit=contains(visible, state['gold']),
                truncated=action['observation_truncated']))
        e = dict(qid=row['qid'], branch=row['branch'], replicate=row['replicate'], em=row['em'],
                 final_answer=row['final_answer'], original_observation_gold_hit=contains('\n'.join(original_observations), state['gold']),
                 any_new_visible_gold_hit=any(x['visible_gold_hit'] for x in observations),
                 observations=observations)
        evidence_rows.append(e)
        examples += [f"## {row['qid']} / {row['branch']} / seed replicate {row['replicate']}", '',
                     f"Question: {state['question']}", f"Gold: {state['gold']}",
                     f"Answer: {row['final_answer']!r}; EM={row['em']}; stop={row['stop_reason']}", '']
        for obs in observations:
            examples += [f"Query: {obs['query']}", f"Gold occurrence: raw={obs['raw_gold_hit']}, visible={obs['visible_gold_hit']}", '',
                         '```text', obs['visible_text'], '```', '']
    grouped = []
    n = manifest['replicates']
    report = ['# Independent retrieval fork results', '',
              f"{len(manifest['state_ids'])} questions, {len(rows)} continuations; all planned keys present once. Frozen strict EM unchanged.", '',
              '| Question | Answer correct | Search correct | Search calls, mean | Invalid search continuations | New visible gold hit |',
              '|---|---:|---:|---:|---:|---:|']
    for sid in manifest['state_ids']:
        a = [r for r in rows if r['state_id'] == sid and r['branch'] == 'answer']
        s = [r for r in rows if r['state_id'] == sid and r['branch'] == 'search']
        e = [r for r in evidence_rows if r['qid'] == states[sid]['qid'] and r['branch'] == 'search']
        item = dict(qid=states[sid]['qid'], replicates=n, answer_correct=sum(r['em'] for r in a),
                    search_correct=sum(r['em'] for r in s), delta_em=(sum(r['em'] for r in s)-sum(r['em'] for r in a))/n,
                    mean_search_calls=statistics.mean(r['search_calls'] for r in s),
                    invalid_search=sum(r['stop_reason']=='invalid_or_incomplete' for r in s),
                    new_visible_gold_hit=sum(r['any_new_visible_gold_hit'] for r in e),
                    canonical_answer_probability=a[0]['preference']['answer_probability'])
        grouped.append(item)
        report.append(f"| {item['qid']} | {item['answer_correct']}/{n} | {item['search_correct']}/{n} | {item['mean_search_calls']:.2f} | {item['invalid_search']}/{n} | {item['new_visible_gold_hit']}/{n} |")
    telemetry = [json.loads(l) for l in (out / 'gpu_watchdog.jsonl').read_text().splitlines()]
    audit = dict(state='complete', n_questions=len(grouped), n_branches=len(rows), unique_expected_keys=True,
                 collector_guard_states_retriever_hashes_match=True, checkpoint_size_mtime_unchanged=True,
                 raw_sha256=sha(out/'branches.jsonl'), report_script_sha256=sha(Path(__file__)),
                 max_sampled_own_gpu_mib=max(t['own_mib'] for t in telemetry),
                 min_sampled_card_free_mib=min(t['card_free_mib'] for t in telemetry),
                 branch_wall_seconds=sum(r['seconds'] for r in rows),
                 stop_reasons=dict(collections.Counter(r['stop_reason'] for r in rows)),
                 scope='Exploratory fixed-state continuations; not training, not an ER-versus-OPD comparison. GPU memory is sampled, not an isolation guarantee.')
    report += ['', f"Total branch time: {audit['branch_wall_seconds']/60:.1f} minutes (includes retrieval waits).",
               f"Sampled own GPU memory max: {audit['max_sampled_own_gpu_mib']} MiB; sampled card free min: {audit['min_sampled_card_free_mib']} MiB.", '',
               'Gold occurrence is a normalized complete-string match with token boundaries. No aliases are added. A missing occurrence is not evidence insufficiency; an occurrence is not causal proof of usable evidence.',
               'The four question panel, if used, cannot establish a reliable state-dependent gain or generalization. Replicate seeds are not independent questions. The free search query distribution is not a fixed DAS candidate query.', '']
    (out/'COMPLETION_AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n')
    (out/'descriptive_table.json').write_text(json.dumps(grouped, indent=2)+'\n')
    (out/'visible_evidence_audit.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False)+'\n' for e in evidence_rows))
    (out/'QUALITATIVE_EXAMPLES.md').write_text('\n'.join(examples))
    (out/'INDEPENDENT_RESULTS.md').write_text('\n'.join(report))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
