"""CPU-only descriptive progress. Never compare different partially sampled sets."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/recovery_continue_teacher_gpu4_20260917'


def read_rows(path):
    if not path.exists():return [],None
    raw=path.read_bytes()
    # A writer may be halfway through its final record at this instant.
    raw=raw[:raw.rfind(b'\n')+1]
    return [json.loads(line) for line in raw.splitlines()],hashlib.sha256(raw).hexdigest()


def main():
    paths=dict(student=ROOT/'reports/recovery_continue_gpu4_20260917/continue_student/continuations.jsonl',
               teacher=OUT/'continue_teacher/continuations.jsonl')
    report=dict(time=time.strftime('%Y-%m-%d %H:%M:%S%z'),scope='INTERIM descriptive only; collection incomplete; no efficacy claim',arms={})
    by_model={}
    for name,path in paths.items():
        rows,digest=read_rows(path);by=defaultdict(list)
        keys={(r['qid'],r['replicate']) for r in rows}
        if len(keys)!=len(rows):raise ValueError('Duplicate keys')
        for r in rows:by[r['qid']].append(r)
        complete={q:rs for q,rs in by.items() if {r['replicate'] for r in rs}==set(range(4))}
        report['arms'][name]=dict(completed=len(rows),total=512,em_accepted=sum(r['em'] for r in rows),
            complete_questions=len(complete),complete_questions_with_success=sum(any(r['em'] for r in rs) for rs in complete.values()),
            stop_reasons=dict(Counter(r['stop_reason'] for r in rows)),source=str(path),complete_lines_sha256=digest)
        by_model[name]=complete
    common=sorted(set(by_model['student'])&set(by_model['teacher']))
    report['paired_complete_questions']=[dict(qid=q,student_correct=sum(r['em'] for r in by_model['student'][q]),
        teacher_correct=sum(r['em'] for r in by_model['teacher'][q])) for q in common]
    report['interpretation']='Only fully sampled common questions are paired; acquisition-order subset is not a final benchmark. EM acceptance does not certify evidence or training usefulness.'
    out=OUT/'interim_progress.json';tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
