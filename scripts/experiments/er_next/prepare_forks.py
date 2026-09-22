"""Freeze exploratory post-retrieval states from ALL 64 existing natural questions.

No outcome filtering. Require an exact token cut before the first action after
the first retrieval. Coverage failures are recorded, never silently filled.
"""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'reports/evidence_stopping_gpu5_20260915'
OUT = ROOT / 'reports/er_next_20260916'


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ROOT / 'data/student/0.5B', local_files_only=True)
    panel = json.loads((SOURCE / 'natural_panel.json').read_text())
    path = SOURCE / 'trajectories.jsonl'
    records = [json.loads(l) for l in path.read_text().splitlines()]
    labels = sorted({r['label'] for r in records})
    label = next((s for s in labels if 'opd' in s.lower() and 'er' not in s.lower() and '50' in s), None)
    assert label is not None, labels
    by_q = {r['qid']: r for r in records if r['label'] == label and r['condition'] == 'natural'}
    assert len(by_q) == len(panel) == 64
    states, coverage = [], []
    for q in sorted(panel, key=lambda q: q['qid']):
        ids = tok.apply_chat_template(q['prompt'], tokenize=True, add_generation_prompt=True)
        retrieved = 0
        reason = 'no_post_retrieval_decision'
        for turn, action in enumerate(by_q[q['qid']]['actions']):
            if retrieved:
                g = action['generation']; text = g['raw_text']
                opening = re.search(r'<(?:search|answer)>', text)
                if opening is None:
                    reason = 'missing_action_opening'; break
                pre = text[:opening.start()]
                if '</think>' not in pre:
                    reason = 'missing_completed_think'; break
                cuts = [i for i in range(len(g['token_ids']) + 1)
                        if tok.decode(g['token_ids'][:i], skip_special_tokens=False) == pre]
                if not cuts:
                    reason = 'no_exact_token_boundary'; break
                state_ids = ids + g['token_ids'][:cuts[0]]
                assert action['context_length'] == len(ids)
                if len(state_ids) + 259 > 4096:
                    reason = 'insufficient_context'; break
                states.append(dict(state_id=q['qid'] + ':post1', qid=q['qid'],
                    question=q['question'], gold=q['gold'], prefix_ids=state_ids,
                    prefix_text=tok.decode(state_ids, skip_special_tokens=False),
                    source_label=label, turn=turn, prior_search_calls=retrieved,
                    origin='previously_inspected_exploratory_panel',
                    native_action=action['action']))
                reason = 'included'; break
            ids += action['generation']['token_ids']
            if 'observation_ids' in action:
                ids += action['observation_ids']; retrieved += 1
        coverage.append(dict(qid=q['qid'], reason=reason))
    payload = dict(schema_version=1, exploratory=True, population_questions=64,
        source_label=label, states=states, coverage=coverage,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [path, SOURCE / 'natural_panel.json', Path(__file__)]})
    target = OUT / 'fork_states.frozen.json'
    if target.exists():
        assert json.loads(target.read_text()) == payload, 'Frozen states changed; use a new run directory.'
    else:
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(included=len(states), population=64,
        coverage=__import__('collections').Counter(r['reason'] for r in coverage)), ensure_ascii=False))


if __name__ == '__main__':
    main()
