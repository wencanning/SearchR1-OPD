"""All predefined metrics for the narrow-claim CPU probe."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/claim_intervention_cpu_20260914'
CONDITIONS = ['original','hide_claims','delete_claims','hide_matched_claims','hide_matched_benign','hide_all_post']
METRICS = ['full_sum','answer_sum','answer_mean','full_mean']


def audit_inputs(manifest):
    """Verify the frozen input contract without rerunning model inference."""
    audit = []
    for case in manifest['cases']:
        conditions = case['conditions']
        original = conditions['original']
        assert hashlib.sha256((ROOT/case['source_file']).read_bytes()).hexdigest() == case['source_sha256']
        assert set(conditions) == set(CONDITIONS)
        assert all(original['mask'])
        for name, condition in conditions.items():
            assert len(condition['ids']) == len(condition['mask'])
            if name != 'delete_claims':
                assert condition['ids'] == original['ids']
        hidden = {i for i, visible in enumerate(conditions['hide_claims']['mask']) if not visible}
        annotated = {i for span in case['annotated_spans'] for i in span['indices']}
        broad = {i for i, visible in enumerate(conditions['hide_all_post']['mask']) if not visible}
        matched = {i for i, visible in enumerate(conditions['hide_matched_claims']['mask']) if not visible}
        benign = {i for i, visible in enumerate(conditions['hide_matched_benign']['mask']) if not visible}
        assert hidden == annotated and hidden < broad
        assert matched <= hidden and benign.isdisjoint(broad)
        assert len(matched) == len(benign) == case['matched_tokens']
        assert len(hidden) == case['claim_tokens']
        assert conditions['delete_claims']['ids'] == [v for i,v in enumerate(original['ids']) if i not in hidden]
        assert all(conditions['delete_claims']['mask'])
        audit.append(dict(key=case['key'],source_hash_verified=True,
            claim_tokens=len(hidden),matched_tokens=len(matched),retained_post_tokens=len(broad-hidden)))
    return audit


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    input_checks=audit_inputs(manifest)
    rows=[json.loads(l) for l in (OUT/'scores.jsonl').read_text().splitlines()]
    excluded_path=OUT/'excluded.jsonl'
    excluded=[json.loads(l) for l in excluded_path.read_text().splitlines()] if excluded_path.exists() else []
    assert len(rows)==len({r['key'] for r in rows})
    assert len(rows)+len(excluded)==len(manifest['cases']), 'Incomplete probe'
    assert {r['key'] for r in rows+excluded} == {c['key'] for c in manifest['cases']}
    flat=[]; max_prefix_error=0; max_duplicate_error=0
    for r in rows:
        assert set(r['scores'])==set(CONDITIONS)
        assert r['baseline_max_candidate_error']<.5
        if r['matched_is_full']:
            for candidate in ['gold','wrong']:
                left=r['scores']['hide_claims'][candidate]
                right=r['scores']['hide_matched_claims'][candidate]
                assert left['token_ids']==right['token_ids']
                error=max(abs(a-b) for a,b in zip(left['token_logps'],right['token_logps']))
                max_duplicate_error=max(max_duplicate_error,error)
                assert error<1e-5, 'Repeated identical condition is not numerically reproducible'
        for c,s in r['scores'].items():
            assert s['gold']['first_top_token']==s['wrong']['first_top_token']
            diff=abs(s['gold']['first_top_logp']-s['wrong']['first_top_logp'])
            max_prefix_error=max(max_prefix_error,diff)
            assert diff<1e-3
            flat.append(dict(key=r['key'],group=r['group'],condition=c,
                first_top_token=s['gold']['first_top_token'],
                **{metric:s['gold'][metric]-s['wrong'][metric] for metric in METRICS}))
    with (OUT/'margins.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0])); writer.writeheader();writer.writerows(flat)
    counts={}
    for group in sorted({r['group'] for r in rows}):
        counts[group]={}
        for c in CONDITIONS:
            selected=[r for r in flat if r['condition']==c and r['group']==group]
            counts[group][c]=dict(n=len(selected),**{m:sum(r[m]>0 for r in selected) for m in METRICS})
    report=['# 原生结尾、窄范围断言干预：全部指标','',
        '正值更支持正确候选，负值更支持错误候选。单位nat；不是自由生成准确率。','',
        'hide_claims/delete_claims=全部标注断言；matched条件=等量完整断言子集及开头复述；hide_all_post=广范围参照。',
        '正确对照的标注是正确断言；失败案例的等量断言条件不覆盖全部错误。','']
    for metric in METRICS:
        report += [f'## {metric}','','|前缀|'+'|'.join(CONDITIONS)+'|','|---|'+'---:|'*len(CONDITIONS)]
        for r in rows:
            values=[next(x for x in flat if x['key']==r['key'] and x['condition']==c)[metric] for c in CONDITIONS]
            report.append('|'+r['key']+'|'+'|'.join(f'{v:.4f}' for v in values)+'|')
        report.append('')
    checks=dict(completed=len(rows),excluded=excluded,forwards=len(rows)*12,
        max_native_candidate_logp_error=max(r['baseline_max_candidate_error'] for r in rows),
        max_prefix_logp_error=max_prefix_error,total_forward_seconds=sum(r['seconds'] for r in rows),
        max_identical_condition_logp_error=max_duplicate_error,
        input_checks=input_checks)
    previous_path=ROOT/'reports/history_intervention_cpu_20260914/scores.jsonl'
    if previous_path.exists():
        previous={r['key']:r for line in previous_path.read_text().splitlines() for r in [json.loads(line)]}
        body_errors=[abs(r['scores'][new][candidate]['answer_sum']-previous[r['key']]['scores'][old][candidate]['answer_sum'])
            for r in rows if r['key'] in previous
            for new,old in [('original','original'),('hide_all_post','hide_post_think')]
            for candidate in ['gold','wrong']]
        checks['cross_closing_body_sum_max_error']=max(body_errors) if body_errors else None
    report += ['## 完成检查','','```json',json.dumps(checks,ensure_ascii=False,indent=2),'```','',
               '## 排序计数','','仅预选五例的候选排序，不用于估计总体发生率。','','```json',json.dumps(counts,ensure_ascii=False,indent=2),'```']
    (OUT/'RESULTS.md').write_text('\n'.join(report)+'\n')
    (OUT/'summary.json').write_text(json.dumps(dict(checks=checks,counts=counts,margins=flat),ensure_ascii=False,indent=2))
    print(json.dumps(checks,ensure_ascii=False,indent=2))
    print('\n'.join(report[:14]))


if __name__=='__main__':main()
