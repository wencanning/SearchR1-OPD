"""CPU-only, equal-size support/non-support attention interventions."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/evidence_support_access_cpu_20260916'
sp = importlib.util.spec_from_file_location('fixed_history_helpers', ROOT / 'scripts/experiments/evidence_history_control_cpu/worker.py')
H = importlib.util.module_from_spec(sp); sp.loader.exec_module(H)
P, M = H.P, H.M


def runs(indices):
    result = []
    for i in sorted(indices):
        if result and result[-1][-1] + 1 == i:
            result[-1].append(i)
        else:
            result.append([i])
    return result


def matched_controls(pool, support, qid):
    lengths = sorted((len(x) for x in runs(support)), reverse=True)
    rng = random.Random('support-control-20260916|' + qid)
    chosen = set()
    for _ in range(1000):
        available = set(pool); pick = []
        for n in lengths:
            windows = [run[i:i+n] for run in runs(available) for i in range(len(run)-n+1)]
            if not windows:
                break
            w = rng.choice(windows); pick += w
            # Keep distinct spans distinct, preserving the support run geometry.
            available -= set(w) | {w[0]-1, w[-1]+1}
        else:
            chosen.add(tuple(sorted(pick)))
        if len(chosen) == 3:
            return [list(x) for x in sorted(chosen)]
    return None


def build_masks(tok, q, aliases):
    ids, _ = M.make_prefix(tok, q, 'both_hops')
    text = tok.decode(ids, skip_special_tokens=False)
    info_start, info_end = list(re.finditer(r'<information>(.*?)</information>', text, re.S))[-1].span(1)
    body = q['second']['text']
    body_start = text.rfind(body)
    assert info_start <= body_start < info_end and body_start + len(body) <= info_end
    support_spans = []
    for sentence in q['second_support']:
        hits = list(re.finditer(re.escape(sentence), body))
        assert len(hits) == 1, 'Ambiguous annotated support sentence'
        support_spans.append((body_start + hits[0].start(), body_start + hits[0].end()))
    protected = support_spans.copy()
    for answer in set(q['gold'] + aliases):
        if answer.strip():
            for hit in re.finditer(r'(?<!\w)' + re.escape(answer) + r'(?!\w)', body, re.I):
                protected.append((body_start + hit.start(), body_start + hit.end()))
    lengths = [0] + [len(tok.decode(ids[:i], skip_special_tokens=False)) for i in range(1, len(ids)+1)]
    offsets = list(zip(lengths, lengths[1:]))
    def overlaps(a, b, spans):
        return any(a < end and b > start for start, end in spans)
    evidence = [i for i, (a,b) in enumerate(offsets) if b > a and a >= info_start and b <= info_end]
    body_indices = [i for i,(a,b) in enumerate(offsets) if b > a and a >= body_start and b <= body_start+len(body)]
    support = [i for i in body_indices if overlaps(*offsets[i], support_spans)]
    pool = [i for i in body_indices if not overlaps(*offsets[i], protected)]
    if not support:
        return None, dict(reason='No eligible support tokens')
    control = matched_controls(pool, support, q['qid'])
    if control is None:
        return None, dict(reason='Fewer than three distinct same-document controls matching support run lengths',
                          support_tokens=len(support), non_support_tokens=len(pool), support_run_lengths=[len(x) for x in runs(support)])
    geometry = sorted(len(x) for x in runs(support))
    for ix in control:
        assert len(ix) == len(support) and not set(ix) & set(support)
        assert sorted(len(x) for x in runs(ix)) == geometry
    masks = {'all_hidden': evidence, 'intact': [], 'support_hidden': support}
    masks.update({f'control_{i+1}_hidden': ix for i,ix in enumerate(control)})
    artifact = dict(prefix_ids=ids, masks=masks, support_spans=support_spans,
                    masked_text={name: [tok.decode([ids[j] for j in run], skip_special_tokens=False) for run in runs(ix)]
                                 for name,ix in masks.items() if name not in ['all_hidden','intact']},
                    support_run_lengths=geometry, support_tokens=len(support), non_support_pool=len(pool))
    return artifact, dict(reason='eligible', support_tokens=len(support), non_support_tokens=len(pool))


def prepare(tok):
    source = P.SOURCE
    protocol = M.read(source / 'protocol.frozen.json')
    excluded = set(M.read(P.OUT/'inputs.frozen.json')['selected_teacher_qids']) | set(M.read(H.OUT/'inputs.frozen.json')['selected_qids'])
    qmap = {q['qid']:q for q in M.read(source/'candidates.json')}
    review = {q['qid']:q for q in M.read(source/'review.json')}
    original = {(r['label'],r['qid'],r['condition']):r for r in P.rows(source/'first_actions.jsonl')}
    ordered = sorted(set(protocol['selected_qids'])-excluded,
                     key=lambda q: hashlib.sha256(('support-access-20260916|'+q).encode()).hexdigest())
    selection = []; chosen = []; cases = []; reviews = []
    for qid in ordered:
        item, eligibility = build_masks(tok, qmap[qid], review[qid]['aliases'])
        selected = item is not None and len(chosen) < 8
        selection.append(dict(qid=qid, selected=selected, **eligibility))
        if not selected:
            continue
        chosen.append(qid)
        record = original['opd-step100',qid,'first_hop']
        boundary = P.make_boundary(tok,record,'')
        assert boundary is not None, 'Do not replace question based on missing model boundary'
        actual = boundary['ids'][len(record['input_ids']):]
        reviews.append(dict(qid=qid,question=qmap[qid]['question'],second_document=qmap[qid]['second'],
                            support_sentences=qmap[qid]['second_support'],gold=qmap[qid]['gold'],**item))
        for history,hids in [('neutral',tok.encode(H.NEUTRAL,add_special_tokens=False)),('fixed_student',actual)]:
            ids=item['prefix_ids']+hids
            masks={name:[int(i in set(indices)) for i in range(len(ids))] for name,indices in item['masks'].items()}
            assert len(ids)+8<4096
            cases.append(dict(qid=qid,history=history,ids=ids,masks=masks,history_ids=hids,
                              support_tokens=item['support_tokens'],support_run_lengths=item['support_run_lengths']))
    assert len(chosen)==8, ('Too few input-eligible questions; stop before inference',selection)
    sources=[source/n for n in ['protocol.frozen.json','candidates.json','review.json','first_actions.jsonl']]
    sources += [P.OUT/'inputs.frozen.json',H.OUT/'inputs.frozen.json',Path(P.__file__),Path(H.__file__),Path(M.__file__),
                ROOT/'verl/utils/evidence_residual.py',Path(__file__)]
    manifest=dict(cases=cases, selected_qids=chosen, excluded_qids=sorted(excluded), input_selection=selection,
                  alphas=[0,.5,1,1.5],tags={a:tok.encode('<'+a+'>',add_special_tokens=False) for a in ['answer','search']},
                  source_sha256={str(p.relative_to(ROOT)):H.sha(p) for p in sources},
                  environment='evidence-stopping-cpu@a33f1b1d',device='cpu',dtype='float32',threads=4,
                  design='Same input IDs and positions; support versus 3 same-document non-support masks, equal counts and contiguous run lengths. All-hidden reference fixed across interventions.',
                  caveat='Artificial attention intervention, not ordinary retrieved-text replacement. ER amplification under fixed hidden logits partly follows algebra.')
    path=OUT/'inputs.frozen.json'
    if path.exists():
        assert M.read(path)==manifest, 'Frozen input/code mismatch'
    else:
        M.save(path,manifest)
        M.save(OUT/'MASK_REVIEW.json',reviews)
    return manifest


def run(manifest,torch):
    from transformers import AutoModelForCausalLM
    from verl.utils.evidence_residual import build_evidence_hidden_attention_mask
    path=OUT/'teacher_paths.jsonl';old=P.rows(path)
    done={(r['qid'],r['history'],r['action']) for r in old};assert len(old)==len(done)
    model=AutoModelForCausalLM.from_pretrained(P.TEACHER,local_files_only=True,torch_dtype=torch.float32,
            attn_implementation='eager',low_cpu_mem_usage=True).eval().to('cpu')
    assert all(p.device.type=='cpu' for p in model.parameters())
    print('LOAD_CPU_TEACHER_COMPLETE',flush=True)
    def forward(ids,mask,keep):
        x=torch.tensor([ids],device='cpu');valid=torch.ones_like(x)
        attention=valid if mask is None else build_evidence_hidden_attention_mask(valid,torch.tensor([mask]),torch.float32)
        with torch.inference_mode():
            z=model(x,attention_mask=attention,position_ids=torch.arange(len(ids))[None,:],use_cache=False,num_logits_to_keep=keep).logits[0].float()
        assert z.shape[0]==keep and torch.isfinite(z).all() and not torch.cuda.is_initialized()
        return z
    def score(z,tag):
        lp=z.log_softmax(-1);values=lp[torch.arange(len(tag)),torch.tensor(tag)]
        return dict(log_probability=float(values.sum()),token_log_probabilities=values.tolist())
    for c in manifest['cases']:
        for action,tag in manifest['tags'].items():
            key=(c['qid'],c['history'],action)
            if key in done:continue
            tic=time.time();ids=c['ids']+tag[:-1];tail=[0]*(len(tag)-1)
            hidden=forward(ids,c['masks']['all_hidden']+tail,len(tag))
            variants={}
            for name,mask in c['masks'].items():
                if name=='all_hidden':continue
                z=forward(ids,None if name=='intact' else mask+tail,len(tag))
                if name=='intact' and not(OUT/'NULL_MASK_CHECK.json').exists():
                    zero=forward(ids,[0]*len(ids),len(tag));error=float((z-zero).abs().max())
                    aligned=forward(c['ids'],None,1);alignment=float((z[0]-aligned[0]).abs().max())
                    assert error<1e-4 and alignment<1e-3,(error,alignment)
                    M.save(OUT/'NULL_MASK_CHECK.json',dict(passed=True,max_error=error,causal_alignment_max_error=alignment,cuda_initialized=False))
                variants[name]={str(a):score(z+a*(z-hidden),tag) for a in manifest['alphas']}
                print('CPU_VARIANT',*key,name,flush=True)
            row=dict(qid=c['qid'],history=c['history'],action=action,variants=variants,hidden=score(hidden,tag),
                     support_tokens=c['support_tokens'],seconds=time.time()-tic,device='cpu',dtype='float32',cuda_initialized=False)
            P.append(path,row);done.add(key)
            M.save(OUT/'status.json',dict(state='running',completed_paths=len(done),total_paths=32,cuda_initialized=False))
            print('CPU_PATH',len(done),32,*key,round(row['seconds'],2),flush=True)
    assert len(P.rows(path))==32
    M.save(OUT/'status.json',dict(state='inference_complete',completed_paths=32,total_paths=32,cuda_initialized=False))


def main():
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20260916)
    assert not torch.cuda.is_initialized()
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    tok=AutoTokenizer.from_pretrained(ROOT/'data/student/0.5B',local_files_only=True)
    manifest=prepare(tok);print('PREPARED',len(manifest['cases']),'states',manifest['selected_qids'],flush=True)
    if args.stage=='run':run(manifest,torch)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        M.save(OUT/'failure.json',dict(error=repr(exc),at=time.strftime('%Y-%m-%d %H:%M:%S')));raise
