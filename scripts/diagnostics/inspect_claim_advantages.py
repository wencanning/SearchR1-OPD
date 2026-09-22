"""Read saved training signals on pre-annotated spans; no model forward."""
import gzip
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/claim_intervention_cpu_20260914'


def main():
    manifest=json.loads((OUT/'manifest.json').read_text()); result=[]
    for c in manifest['cases']:
        with gzip.open(ROOT/c['source_file'],'rt') as f:
            next(f); r=next(r for line in f for r in [json.loads(line)] if int(r['index'])==c['index'])
        pos=0;cut=0
        for token in r['token_texts']:
            if pos>=len(c['prefix_text']):break
            pos+=len(token);cut+=1
        offset=len(c['conditions']['original']['ids'])-cut
        clauses=[]
        for p in c['annotated_spans']:
            ids=[i-offset for i in p['indices']]
            gaps=[r['opd_advantage'][i] for i in ids]
            clauses.append(dict(text=p['text'],n=len(ids),advantage_sum=sum(gaps),advantage_mean=statistics.mean(gaps),
                negative_tokens=sum(v<0 for v in gaps),weighted_sum=sum(r['weighted_opd_advantage'][i] for i in ids),
                minimum=dict(token=r['token_texts'][ids[gaps.index(min(gaps))]],advantage=min(gaps))))
        # Report the whole observed answer, not only a selected positive first token.
        answer_end=len(c['prefix_text'])+len(c['leading'])+len(c['observed'])
        answer_ids=[]; cursor=len(c['prefix_text'])
        for i in range(cut,len(r['token_texts'])):
            if cursor>=answer_end:break
            answer_ids.append(i);cursor+=len(r['token_texts'][i])
        answer_adv=[r['opd_advantage'][i] for i in answer_ids]
        result.append(dict(key=c['key'],annotation_kind=c['annotation_kind'],clauses=clauses,
            answer_first_token=r['token_texts'][cut],answer_first_advantage=r['opd_advantage'][cut],
            answer_first_effective_coef=r['opd_effective_distillation_coef'][cut],
            answer_body=dict(tokens=[r['token_texts'][i] for i in answer_ids],
                advantages=answer_adv,advantage_sum=sum(answer_adv),advantage_mean=statistics.mean(answer_adv),
                weighted_sum=sum(r['weighted_opd_advantage'][i] for i in answer_ids))))
    (OUT/'saved_claim_advantages.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for c in result:
        print(c['key'],'clause sums',[round(p['advantage_sum'],3) for p in c['clauses']],
              'answer first advantage',round(c['answer_first_advantage'],3))


if __name__=='__main__':main()
