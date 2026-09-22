"""Summarize completed ordinary teacher demonstrations; no training/effect claim."""
import hashlib
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/recovery_teacher_start_gpu4_20260917'


def main():
    source=ROOT/'reports/recovery_queue_20260917/train_inputs.frozen.json'
    protocol=json.loads(source.read_text());cases={c['qid']:c for c in protocol['cases']}
    status=json.loads((OUT/'question_start/status.json').read_text())
    assert status['state']=='complete' and status['completed']==status['total']==512
    raw=OUT/'question_start/continuations.jsonl';rows=[json.loads(l) for l in raw.read_text().splitlines()]
    expected={(q,k) for q in cases for k in range(4)}
    assert len(rows)==512 and {(r['qid'],r['replicate']) for r in rows}==expected
    spec=importlib.util.spec_from_file_location('scorer',ROOT/'scripts/experiments/evidence_use/experiment.py')
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    for r in rows:
        assert r['split']=='train' and r['model']=='teacher'
        assert r['em']==scorer.exact(r['final_answer'],cases[r['qid']]['gold'])
    passed=[r for r in rows if r['em']]
    (OUT/'em_accepted_teacher_demonstrations.REVIEW_REQUIRED.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in passed))
    summary=dict(state='complete',questions=len(cases),completions=len(rows),em_correct=len(passed),
                 questions_with_success=len({r['qid'] for r in passed}),search_calls=sum(r['search_calls'] for r in rows),
                 generated_tokens=sum(r['generated_tokens'] for r in rows),stop_reasons=dict(Counter(r['stop_reason'] for r in rows)),
                 raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 scope='Question-start teacher demonstrations on TRAIN-only input pool. EM accepted using inherited source references; factual review required before training. Not heldout efficacy, not matched-total-interaction comparison to student-state recovery. No training performed.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
