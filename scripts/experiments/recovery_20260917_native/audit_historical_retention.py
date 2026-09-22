"""CPU-only descriptive outcome transitions; not a causal forgetting test."""
import collections
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'reports/recovery_20260917_native'


def main():
    source=ROOT/'reports/evidence_stopping_gpu5_20260915/trajectories.jsonl'
    protocol=source.with_name('protocol.frozen.json');spec=json.loads(protocol.read_text());assert spec['do_sample'] is False
    rows=[r for r in map(json.loads,source.read_text().splitlines()) if r['condition']=='natural']
    by={}
    for r in rows:
        key=r['label'];assert r['qid'] not in by.setdefault(key,{})
        by[key][r['qid']]=bool(r['final_em'])
    assert all(set(v)==set(spec['natural_qids']) for v in by.values())
    comparisons={}
    for left,right in [('initial-step0','opd-step50'),('opd-step50','opd-step100'),('initial-step0','er-alpha1.5-step50'),('er-alpha1.5-step50','er-alpha1.5-step100')]:
        a,b=by[left],by[right];groups={name:[q for q in a if (a[q],b[q])==pair] for name,pair in [('retained',(True,True)),('lost',(True,False)),('gained',(False,True)),('both_wrong',(False,False))]}
        comparisons[left+' to '+right]=dict(n=len(a),before_correct=sum(a.values()),after_correct=sum(b.values()),counts={k:len(v) for k,v in groups.items()},qids=groups)
    result=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),decoding='greedy',comparisons=comparisons,scope='Post-hoc descriptive64-question heldout natural-agent panel. Correct-to-wrong transitions are observed behavior changes, not evidence that gradient interference or a particular loss caused forgetting. Different checkpoints can change retrieval paths. Do not train on these evaluation questions. No formalSOD/DGPO comparison or independent confirmatory test.')
    (OUT/'HISTORICAL_RETENTION.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Historical success retention audit (CPU only)','',result['scope'],'','|Checkpoints|Before|After|Retained|Lost|New successes|','|---|---:|---:|---:|---:|---:|']
    for name,r in comparisons.items():
        c=r['counts'];lines.append(f"|{name}|{r['before_correct']}/64|{r['after_correct']}/64|{c['retained']}|{c['lost']}|{c['gained']}|")
    lines+=['','This can motivate a future success-retention diagnostic if successful-student replay proves useful. It does not establish that replay solves forgetting, that ER is superior, or that a new method is necessary. Required next controls would separate preservation of prior successes from learning new questions under matched full training; no causal hypothesis is accepted here.']
    (OUT/'HISTORICAL_RETENTION.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v['counts'] for k,v in comparisons.items()}))


if __name__=='__main__':main()
