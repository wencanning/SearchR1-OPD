"""Complete population/semantic/probe report, retaining uncertain and excluded cases."""
import csv
import gzip
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

from audit_pure_opd_mainline import features

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/independent_history_20260914'
TAG='20260914-1220-coef1'


def wilson(k,n):
    if not n:return None
    z=1.95996398454;den=1+z*z/n;center=(k/n+z*z/(2*n))/den
    radius=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/den
    return [center-radius,center+radius]


def summarize(rs):
    n=len(rs);wrong=[r for r in rs if not r['score']];with_answer=[r for r in wrong if r['answer_token_n']]
    return dict(n=n,correct=sum(r['score'] for r in rs),wrong=len(wrong),
        no_answer=sum(r['missing_answer'] for r in rs),repeated_query=sum(r['repeated_query'] for r in rs),
        no_post_think=sum(not r['post_think'] for r in rs),no_post_narrative=sum(not r['post_narrative'] for r in rs),
        mean_policy_tokens=statistics.mean(r['policy_tokens'] for r in rs),
        mean_searches=statistics.mean(r['searches'] for r in rs),
        wrong_with_answer=len(with_answer),wrong_answer_positive_adv=sum(r['answer_adv_sum']>0 for r in with_answer),
        wrong_answer_adv_gt_point1=sum(r['answer_adv_sum']>.1 for r in with_answer),
        malformed_information=sum(r['malformed_information'] for r in rs),
        data_sources=dict(Counter(r['data_source'] for r in rs)))


def main():
    manifest=json.loads((OUT/'sample_manifest.json').read_text());annotations=json.loads((OUT/'annotations.json').read_text())
    assert len(annotations)==50 and len({r['review_id'] for r in annotations})==50
    labels={r['review_id']:r for r in annotations};by_key={r['key']:r for r in manifest['cases']}
    enriched=[];population=[];window_summary=[];source_checks=[]
    for method in ['opd','sod']:
        for lo,hi in [(61,70),(91,100)]:
            window=[]
            for step in range(lo,hi+1):
                path=ROOT/f'verl_checkpoints/pure-{method}-05B-n1-{TAG}/opd_diagnostics/step_{step:06d}.jsonl.gz'
                relative=str(path.relative_to(ROOT));digest=hashlib.sha256(path.read_bytes()).hexdigest()
                if relative in manifest['source_hashes']:assert digest==manifest['source_hashes'][relative]
                with gzip.open(path,'rt') as f:
                    header=json.loads(next(f));rows=[json.loads(l) for l in f]
                assert len(rows)==header['batch_size']==header['selected_sequences']==128
                metrics=header['metrics'];assert metrics['opd/beta']==1 and metrics['opd/grpo_advantage']==0
                assert metrics['opd/teacher_target_is_intervened']==0
                source_checks.append(dict(file=relative,sha256=digest,n=128,beta=1,grpo_advantage=0))
                for r in rows:
                    assert not any(r['grpo_advantage'])
                    key=f'{method}:{step}:{int(r["index"])}';text=''.join(r['token_texts'])
                    f=features(text,r['ground_truth_targets']);assert f['score']==r['sequence_score']
                    first_env=next((i for i,v in enumerate(r['loss_mask']) if not v),len(r['loss_mask']))
                    post_think=any(i>first_env and m and s=='think' and t.strip() for i,(m,s,t) in
                        enumerate(zip(r['loss_mask'],r['segments'],r['token_texts'])))
                    narrative_ids=[i for i,(m,s,t) in enumerate(zip(r['loss_mask'],r['segments'],r['token_texts']))
                        if i>first_env and m and s not in ['tag','search','answer','after_answer'] and t.strip()]
                    answer_ids=[i for i,(m,s) in enumerate(zip(r['loss_mask'],r['segments'])) if m and s=='answer']
                    row=dict(key=key,method=method,step=step,index=r['index'],data_source=r['data_source'],
                        **f,post_think=bool(post_think),post_narrative=bool(narrative_ids),policy_tokens=sum(r['loss_mask']),
                        answer_token_n=len(answer_ids),answer_adv_sum=sum(r['opd_advantage'][i] for i in answer_ids),
                        answer_weighted_adv_sum=sum(r['weighted_opd_advantage'][i] for i in answer_ids))
                    window.append(row);population.append(row)
                    if key in by_key:
                        c=by_key[key];a=labels[c['review_id']]
                        assert text==c['response']
                        annotation_signals=[];bounds=[];pos=0
                        for t in r['token_texts']:bounds.append((pos,pos+len(t)));pos+=len(t)
                        for snippet in a['spans']:
                            assert text.count(snippet)==1
                            start=text.index(snippet);end=start+len(snippet)
                            ids=[i for i,(s,e) in enumerate(bounds) if s<end and e>start]
                            annotation_signals.append(dict(text=snippet,n_tokens=len(ids),policy_tokens=sum(r['loss_mask'][i] for i in ids),
                                segments=dict(Counter(r['segments'][i] for i in ids)),
                                raw_adv_sum=sum(r['opd_advantage'][i] for i in ids if r['loss_mask'][i]),
                                weighted_adv_sum=sum(r['weighted_opd_advantage'][i] for i in ids if r['loss_mask'][i])))
                        enriched.append(dict(**row,**a,question=c['question'],gold=c['gold'],annotation_signals=annotation_signals))
            window_summary.append(dict(method=method,window=[lo,hi],**summarize(window)))
            print('WINDOW',method,lo,hi,flush=True)
    assert len(enriched)==50
    enriched.sort(key=lambda r:r['review_id'])
    semantic={}
    for method in ['opd','sod']:
        semantic[method]={}
        for score,label in [(0,'em_wrong'),(1,'em_correct')]:
            rs=[r for r in enriched if r['method']==method and r['score']==score]
            assert len(rs)==(20 if score==0 else 5)
            joint=sum(r['evidence']=='sufficient' and r['answer_semantic']=='wrong' and r['history_error']=='yes' for r in rs)
            semantic[method][label]=dict(n=len(rs),answer_semantic=dict(Counter(r['answer_semantic'] for r in rs)),
                evidence=dict(Counter(r['evidence'] for r in rs)),history_error=dict(Counter(r['history_error'] for r in rs)),
                sufficient_wrong_history_error=joint,joint_wilson95=wilson(joint,len(rs)),
                no_post_narrative=sum(not r['post_narrative'] for r in rs),
                sufficient_no_answer=sum(r['evidence']=='sufficient' and r['missing_answer'] for r in rs),
                definite_wrong_positive_answer_adv=sum(r['answer_semantic']=='wrong' and r['answer_token_n']>0 and r['answer_adv_sum']>0 for r in rs),
                definite_wrong_with_answer=sum(r['answer_semantic']=='wrong' and r['answer_token_n']>0 for r in rs))
    probe_manifest=json.loads((OUT/'probe_manifest.json').read_text())
    assert probe_manifest['annotations_sha256']==hashlib.sha256((OUT/'annotations.json').read_bytes()).hexdigest()
    assert probe_manifest['sample_sha256']==hashlib.sha256((OUT/'sample_manifest.json').read_bytes()).hexdigest()
    scores=[json.loads(l) for l in (OUT/'probe_scores.jsonl').read_text().splitlines()]
    numerical_path=OUT/'numerical_exclusions.jsonl'
    numerical=[json.loads(l) for l in numerical_path.read_text().splitlines()] if numerical_path.exists() else []
    assert {r['key'] for r in scores+numerical}=={r['key'] for r in probe_manifest['cases']}
    assert len(scores)+len(numerical)==len(probe_manifest['cases'])
    metrics=['full_sum','answer_sum','answer_mean','full_mean'];flat=[]
    for r in scores:
        assert r['baseline_max_error']<.5
        assert set(r['scores'])=={'original','hide_claims','hide_all_post'}
        for condition,s in r['scores'].items():
            flat.append(dict(key=r['key'],review_id=r['review_id'],condition=condition,device=r['device'],
                **{metric:s['gold'][metric]-s['wrong'][metric] for metric in metrics}))
    generation=[json.loads(l) for l in (OUT/'free_generation.jsonl').read_text().splitlines()]
    expected={(c['key'],condition) for c in probe_manifest['cases'] if c['free_generate'] and c['key'] not in {r['key'] for r in numerical} for condition in ['original','hide_claims']}
    assert {(r['key'],r['condition']) for r in generation}==expected and len(generation)==len(expected)
    summary=dict(windows=window_summary,semantic=semantic,probe=dict(n=len(scores),numerical_exclusions=numerical,
        design_exclusions=probe_manifest['excluded'],max_baseline_error=max(r['baseline_max_error'] for r in scores),
        candidate_forward_seconds=sum(r['seconds'] for r in scores),generation=generation,margins=flat),source_checks=source_checks)
    (OUT/'analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (OUT/'review_with_signals.json').write_text(json.dumps(enriched,ensure_ascii=False,indent=2))
    (OUT/'window_features.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in population))
    with (OUT/'probe_margins.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    review=['# 全部匿名样本的语义标注与方法揭示','','语义标注先冻结，随后才合并方法、EM和教师信号。uncertain不强行计作错误。','',
            '|ID|方法/步/index|EM|语义|证据|错误解释|问题分类|','|---|---|---:|---|---|---|---|']
    for r in enriched:review.append(f'|{r["review_id"]}|{r["key"]}|{r["score"]}|{r["answer_semantic"]}|{r["evidence"]}|{r["history_error"]}|{r["error_type"]}|')
    for r in enriched:
        review += ['',f'## {r["review_id"]}: {r["question"]}','','参考：'+str(r['gold']),'','原答案：'+str(r['answer']),'',r['notes'],'',
                   '材料片段：'+str(r['evidence_quotes']),'','标注错误片段：'+str(r['spans'])]
    (OUT/'REVIEW.md').write_text('\n'.join(review)+'\n')
    result=['# 全部预设计分口径','','正数更支持参考候选；负数更支持原错误候选。','']
    for metric in metrics:
        result += [f'## {metric}','','|ID|原始|窄范围屏蔽|全部搜索后think屏蔽|','|---|---:|---:|---:|']
        for r in scores:
            values=[r['scores'][c]['gold'][metric]-r['scores'][c]['wrong'][metric] for c in ['original','hide_claims','hide_all_post']]
            result.append('|'+r['review_id']+'|'+'|'.join(f'{v:.6f}' for v in values)+'|')
        result.append('')
    (OUT/'RESULTS.md').write_text('\n'.join(result)+'\n')
    print(json.dumps(dict(windows=window_summary,semantic=semantic,probe=summary['probe']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
