"""Narrow claim ablation with native answer closing and count-matched controls."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
import argparse
import gzip
import json
import re
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from probe_history_cpu import prepare as prepare_history, digest, ROOT, MODEL

OUT = ROOT / 'reports/claim_intervention_cpu_20260914'
# post-think index (0-based), exact original span. Fixed before scoring.
SPANS = {
    'opd:70:58559': [
        (0, 'that the libretto for "La Statira" was written by Remo Giazotto,'),
        (0, 'based on the opera "Alzira".'),
        (0, 'Therefore, the person who wrote the libretto for "La Statira" is Remo Giazotto.'),
    ],
    'opd:63:23560': [
        (0, 'but I can infer based on the context that this is likely more than the members of The Used.'),
        (1, 'the non-existence of a specific number for Flight of the Conchords and the surrounding context suggest that their membership is more than the members of The Used.'),
        (1, 'Given the options, the one that I would conclude is Flight of the Conchords, as it appears it has more members compared to The Used.'),
    ],
    'sod:67:82035': [
        (0, 'and an intelligence officer in France during World War I,'),
        (0, 'and he was inspired by his idea of Scoutcraft as a hunting technique.'),
        (1, "Frederick Russell Burnham's inspiration in his life was"),
        (1, ', which has been associated with the work of helping the founding of the idea of Scoutcraft as a hunting technique.'),
        (1, 'However, the specific inspiration was not explicitly stated in the provided documents.'),
    ],
    # Correct controls ablate TRUE assertions, not imaginary errors.
    'sod:70:78463': [(0, 'It provides the total number of episodes as 86.')],
    'opd:63:43500': [(0, 'The first armed conflict mentioned was the "first World War."')],
}


def prepare(tok):
    cases = prepare_history(tok)
    for c in cases:
        with gzip.open(ROOT / c['source_file'], 'rt') as f:
            next(f)
            r = next(r for line in f for r in [json.loads(line)] if int(r['index']) == c['index'])
        response = ''.join(r['token_texts'])
        am = list(re.finditer(r'<answer>(.*?)</answer>', response, re.S))[-1]
        end = am.end(1)
        while response[end-1].isspace():
            end -= 1
        c['closing'] = response[end:am.end()]
        assert c['closing'].endswith('</answer>')
        prefix = c['prefix_text']
        thoughts = list(re.finditer(r'<think>(.*?)</think>', prefix, re.S))
        cut = next(i for i in range(len(r['token_texts'])) if ''.join(r['token_texts'][:i]) == prefix)
        prompt_length = len(c['conditions']['original']['ids']) - cut
        # Verify ALL original answer+closing tokens, not merely the first word.
        source_labels = tok(c['leading'] + c['observed'] + c['closing'], add_special_tokens=False)['input_ids']
        assert source_labels == r['token_ids'][cut:cut+len(source_labels)], 'Native answer/closing token mismatch'
        c['saved_original_candidate_logps'] = r['teacher_log_prob'][cut:cut+len(source_labels)]
        bounds, pos = [], 0
        for token in r['token_texts'][:cut]:
            bounds.append((pos, pos+len(token))); pos += len(token)
        pieces = []
        for think_id, snippet in SPANS[c['key']]:
            thought = thoughts[think_id+1]
            body = thought.group(1)
            assert body.count(snippet) == 1
            a = thought.start(1) + body.index(snippet); b = a+len(snippet)
            indices = [prompt_length+i for i,(s,e) in enumerate(bounds)
                       if s<b and e>a and s>=thought.start(1) and e<=thought.end(1) and r['loss_mask'][i]]
            assert indices
            pieces.append(dict(think_id=think_id, text=snippet, indices=indices, start=a, end=b))
        all_indices = sorted({i for p in pieces for i in p['indices']})
        benign = [i for i,m in enumerate(c['conditions']['hide_pre_think_control']['mask']) if not m]
        # Select whole annotated clauses from the end backwards, within benign budget.
        # This is explicitly a subset when all claims cannot be count-matched.
        matched = set(); matched_pieces = []
        for p in reversed(pieces):
            union = matched | set(p['indices'])
            if len(union) <= len(benign):
                matched = union; matched_pieces.append(p['text'])
        assert matched, 'No whole clause fits count-matched budget; do not substitute after seeing scores'
        matched_benign = benign[:len(matched)]
        original = c['conditions']['original']
        def hidden(indices):
            mask = list(original['mask'])
            for i in indices:
                mask[i]=0
            return dict(ids=list(original['ids']), mask=mask)
        removed=set(all_indices)
        ids=[v for i,v in enumerate(original['ids']) if i not in removed]
        broad = c['conditions']['hide_post_think']
        c['conditions'] = dict(original=original, hide_claims=hidden(all_indices),
            delete_claims=dict(ids=ids, mask=[1]*len(ids)),
            hide_matched_claims=hidden(matched), hide_matched_benign=hidden(matched_benign),
            hide_all_post=broad)
        c['annotation_kind'] = 'true_assertion_control' if c['group']=='correct_control' else 'unsupported_or_false_assertions'
        c['annotated_spans'] = pieces
        c['matched_spans'] = matched_pieces
        c['claim_tokens'] = len(all_indices); c['matched_tokens'] = len(matched)
        c['matched_is_full'] = matched == removed
        c['retained_post_tokens'] = c['post_hidden_tokens'] - len(all_indices)
        assert c['retained_post_tokens'] > 0
        assert set(all_indices).isdisjoint(matched_benign)
        assert sum(x==0 for x in c['conditions']['hide_matched_claims']['mask']) == sum(x==0 for x in c['conditions']['hide_matched_benign']['mask'])
        assert all(original['ids'] == value['ids'] for key,value in c['conditions'].items() if key != 'delete_claims')
        # Persist an exact decoded narrow-deletion view for semantic inspection.
        c['deleted_claims_prefix_text'] = tok.decode(ids[prompt_length:], skip_special_tokens=False)
    return cases


@torch.inference_mode()
def evaluate(model, tok, condition, leading, answer, closing):
    encoded = tok(leading+answer+closing, add_special_tokens=False, return_offsets_mapping=True)
    labels = encoded['input_ids']; content_end = len(leading+answer)
    content = [i for i,(a,b) in enumerate(encoded['offset_mapping']) if a<content_end and b>len(leading)]
    x=torch.tensor([condition['ids']+labels[:-1]])
    mask=torch.tensor([condition['mask']+[1]*(len(labels)-1)])
    logits=model(input_ids=x, attention_mask=mask, position_ids=torch.arange(x.shape[1])[None,:],
                 use_cache=False, num_logits_to_keep=len(labels)).logits[0]
    assert torch.isfinite(logits).all()
    lp=logits.log_softmax(-1); chosen=lp.gather(-1,torch.tensor(labels)[:,None]).squeeze(-1)
    top=int(lp[0].argmax()); body=chosen[content]
    return dict(full_sum=chosen.sum().item(), full_mean=chosen.mean().item(), answer_sum=body.sum().item(),
        answer_mean=body.mean().item(), first_logp=chosen[0].item(), first_top_token=tok.decode([top]),
        first_top_logp=lp[0,top].item(), token_ids=labels, token_logps=chosen.tolist(), n_content=len(content), n_full=len(labels))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare-only',action='store_true'); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(42)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    cases=prepare(tok)
    manifest=dict(model=MODEL,torch=torch.__version__,transformers=transformers.__version__,device='cpu',dtype='float32',
                  attention='eager',cache=False,threads=8,cases=cases)
    path=OUT/'manifest.json'
    if path.exists(): assert digest(json.loads(path.read_text()))==digest(manifest)
    else: path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('MANIFEST',digest(manifest),[(c['key'],c['claim_tokens'],c['matched_tokens'],repr(c['closing'])) for c in cases],flush=True)
    if args.prepare_only:return
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32,
        low_cpu_mem_usage=True,attn_implementation='eager').eval()
    assert next(model.parameters()).device.type=='cpu'
    done=set()
    for filename in ['scores.jsonl','excluded.jsonl']:
        p=OUT/filename
        if p.exists():done.update(json.loads(l)['key'] for l in p.read_text().splitlines())
    for c in cases:
        if c['key'] in done:continue
        begin=time.time(); print('START',c['key'],flush=True)
        name='gold' if c['group']=='correct_control' else 'wrong'
        baseline=evaluate(model,tok,c['conditions']['original'],c['leading'],c[name],c['closing'])
        error=abs(baseline['first_logp']-c['baseline_saved_logp'])
        all_errors=[abs(a-b) for a,b in zip(baseline['token_logps'],c['saved_original_candidate_logps'])]
        print('BASELINE',c['key'],error,'max_candidate_error',max(all_errors),flush=True)
        if error>=.5 or max(all_errors)>=.5:
            with (OUT/'excluded.jsonl').open('a') as f:f.write(json.dumps(dict(key=c['key'],reason='native candidate reproduction >= .5 nat',first_error=error,max_error=max(all_errors)))+'\n')
            continue
        result={k:c[k] for k in ['key','group','gold','wrong','closing','claim_tokens','matched_tokens','matched_is_full']}
        result.update(baseline_error=error,baseline_max_candidate_error=max(all_errors),scores={})
        for condition,value in c['conditions'].items():
            scores={}
            for candidate in ['gold','wrong']:
                scores[candidate]=baseline if condition=='original' and candidate==name else evaluate(model,tok,value,c['leading'],c[candidate],c['closing'])
                print('SCORED',c['key'],condition,candidate,'elapsed',round(time.time()-begin,1),flush=True)
            result['scores'][condition]=scores
            print('MARGIN',condition,scores['gold']['full_sum']-scores['wrong']['full_sum'],flush=True)
        result['seconds']=time.time()-begin
        with (OUT/'scores.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
        print('DONE',c['key'],round(result['seconds'],1),flush=True)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
