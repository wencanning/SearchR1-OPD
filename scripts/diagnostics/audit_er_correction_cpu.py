"""Independent artifact checks: recompute distributions and verify frozen inputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/er_correction_cpu_20260914'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def check_panel(folder,alias_only=False):
    assert (folder/'COMPLETED.json').exists()
    m=json.loads((folder/'probe_manifest.json').read_text())
    cases={c['key']:c for c in m['cases']}
    rows=[json.loads(l) for l in (folder/'scores.jsonl').read_text().splitlines()]
    if alias_only:
        r=json.loads((folder/'alias_sensitivity.json').read_text())
        c=next(c for c in m['cases'] if c['review_id']==r['review_id'])
        assert r['post_hoc'] and r['alias']=='Ernest Hemingway'
        rows=[dict(r,key=c['key'],group=c['group'],boundary='native')]
    assert len({(r['key'],r['boundary']) for r in rows})==len(rows)
    counts={'rows':len(rows),'valid':0,'excluded':0,'candidate_recomputations':0}
    max_error=0.;max_entropy_error=0.
    for r in rows:
        c=cases[r['key']]
        assert c['review_id']==r['review_id'] and c['group']==r['group']
        if r['excluded']:
            assert folder==OUT and r['checks']['native_logp_max_error']>=.5
            counts['excluded']+=1;continue
        counts['valid']+=1
        if folder==OUT and r['boundary']=='native':assert r['checks']['native_logp_max_error']<.5
        candidates=['original']+(['gold'] if c['group']!='correct_control' else [])
        for name in candidates:
            path=folder/'logits'/f"{c['review_id']}_{r['boundary']}_{name}{'_alias' if alias_only else ''}.pt"
            if not path.exists():path=OUT/'logits'/path.name
            cache=torch.load(path,map_location='cpu',weights_only=True)
            obs,hid=cache['observed'],cache['hidden']
            # Initial incorrect-vocabulary cache files are never used.
            assert obs.shape==hid.shape and obs.shape[-1]==151665
            labels=torch.tensor(cache['labels']);content=cache['content']
            assert len(labels)==obs.shape[0] and content
            ts=r['scores'][name]['entropy_match_details']
            temps=torch.tensor(ts['temperatures'])
            matched_base=obs-obs.max(-1,keepdim=True).values if 'bracket_expansions' in ts else obs
            methods=dict(observed=obs,hidden=hid,er_0_5=obs+.5*(obs-hid),er_1=obs+(obs-hid),
                         er_1_5=obs+1.5*(obs-hid),temperature_0_7=obs/.7,
                         entropy_matched=matched_base/temps[:,None])
            for method,z in methods.items():
                lp=z.log_softmax(-1);selected=lp.gather(-1,labels[:,None]).squeeze(-1)
                entropy=-(lp.exp()*lp).sum(-1);stored=r['scores'][name][method]
                err=(selected-torch.tensor(stored['token_logps'])).abs().max().item()
                max_error=max(max_error,err)
                assert err<1e-4,(r['review_id'],name,method,err)
                expected=dict(full_sum=selected.sum().item(),full_mean=selected.mean().item(),
                              answer_sum=selected[content].sum().item(),answer_mean=selected[content].mean().item())
                for key,value in expected.items():assert abs(value-stored[key])<1e-4
                assert torch.equal(z.argmax(-1),torch.tensor(stored['top_ids']))
                assert (entropy-torch.tensor(stored['entropy'])).abs().max().item()<1e-4
            actual_gap=(torch.tensor(r['scores'][name]['er_1']['entropy'])-
                        torch.tensor(r['scores'][name]['entropy_matched']['entropy'])).abs().max().item()
            max_entropy_error=max(max_entropy_error,actual_gap)
            assert actual_gap<1e-4
            counts['candidate_recomputations']+=1
        if r['divergence'] is not None:
            x=r['divergence']['log_odds']
            assert abs(x['er_1']-(2*x['observed']-x['hidden']))<1e-4
            assert (x['entropy_matched']>0)==(x['observed']>0)
    from audit_pure_opd_mainline import norm
    generated=[] if alias_only else [json.loads(l) for l in (folder/'generation.jsonl').read_text().splitlines()]
    assert len({(g['key'],g['target']) for g in generated})==len(generated)
    for g in generated:
        assert cases[g['key']]['free_generate'] and g['target'] in ['observed','er_1']
        assert len(g['ids'])<=m['generation_max_tokens']
        if g['stop_reason']=='answer_closed':
            assert g['answer']==g['text'].split('</answer>')[0].strip()
            assert g['reference_em']==(norm(g['answer']) in [norm(v) for v in cases[g['key']]['gold']])
    return dict(**counts,generation_rows=len(generated),max_recomputed_logp_error=max_error,
                max_entropy_gap=max_entropy_error)


def main():
    torch.set_num_threads(1)
    for line in (OUT/'pre_run_hashes.txt').read_text().splitlines():
        digest,rel=line.split(maxsplit=1)
        assert sha(ROOT/rel.strip())==digest,rel
    m=json.loads((OUT/'sample_manifest.json').read_text())
    assert len(m['cases'])==40 and len({c['index'] for c in m['cases']})==40
    assert len({' '.join(c['question'].lower().split()) for c in m['cases']})==40
    assert sum(c['score']==0 for c in m['cases'])==32
    assert not ({c['index'] for c in m['cases']} & set(m['excluded_prior_indices']))
    for rel,digest in m['source_hashes'].items():assert sha(ROOT/rel)==digest
    p=json.loads((OUT/'probe_manifest.json').read_text())
    assert sha(OUT/'sample_manifest.json')==p['sample_sha256']
    assert sha(OUT/'annotations.json')==p['annotations_sha256']
    for c in p['cases']:
        assert len(c['prefix_ids'])==len(c['evidence_mask']) and sum(c['evidence_mask'])>0
        assert max(c['prefix_ids']+c['original_labels'])<151665
    result=dict(source_files_checked=len(m['source_hashes']),unique_review_cases=40,
                frozen_probe_inputs=len(p['cases']),primary=check_panel(OUT),
                fp32_sensitivity=check_panel(OUT/'precision_sensitivity'),cuda_initialized=torch.cuda.is_initialized())
    result['post_hoc_alias']=check_panel(OUT/'precision_sensitivity',alias_only=True)
    assert not result['cuda_initialized']
    (OUT/'AUDIT.json').write_text(json.dumps(result,indent=2))
    (OUT/'CHECKS.md').write_text('# 产物复核\n\nPASS：固定输入与来源哈希、唯一问题抽样、原门槛排除、有效词表、'
        '从原始logits重算所有方法及四种候选分数、熵匹配误差、局部赔率恒等式、预定续写名单与参考EM。\n\n'
        '这是独立脚本的数值重算，由同一主代理完成，不是独立外部研究审稿。'
        '精度敏感性保留未通过历史FP16复现的案例，仅用于同一CPU FP32教师内部的比较。\n\n'
        '```json\n'+json.dumps(result,indent=2)+'\n```\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
