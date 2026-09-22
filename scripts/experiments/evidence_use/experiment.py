"""Fixed-input evidence-use pilot; no training or teacher calls.

Run prepare, freeze the input-only review, then run on physical GPU 5.
Every model sees identical documents and prefixes. All generated actions and
retrieval responses are persisted. This is a diagnostic, not a multi-seed trial.
"""
import argparse
import collections
import hashlib
import html
import json
import os
from pathlib import Path
import random
import re
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/evidence_use_gpu5_20260915'
GPU = 'GPU-2abc6681-2116-879c-a361-749edb32ac3b'
CONDITIONS = ('first_hop', 'evidence_removed', 'with_distractors', 'both_hops')


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def norm(s):
    s = html.unescape(unicodedata.normalize('NFKC', s)).lower()
    return ' '.join(re.sub(r'[^\w\s]', ' ', s).split())


def mention(phrase, text):
    return (' ' + norm(phrase) + ' ') in (' ' + norm(text) + ' ')


def base_title(s):
    return re.sub(r'\s*\([^)]*\)\s*$', '', s).strip()


def exact(answer, gold):
    import string
    def em(s):
        s = ''.join(c for c in s.lower() if c not in string.punctuation)
        return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split())
    return answer is not None and any(em(answer) == em(g) for g in gold)


def doc(title, sentences):
    return dict(title=title, text=''.join(sentences))


def prepare():
    import pandas as pd
    from transformers import AutoTokenizer
    if (OUT / 'panel.json').exists():
        raise RuntimeError('Panel exists; refusing to select new examples after seeing outputs')
    tok = AutoTokenizer.from_pretrained(ROOT / 'data/student/0.5B', local_files_only=True)
    test_path = ROOT / 'data/nq_hotpotqa_train/test.parquet'
    test = pd.read_parquet(test_path)
    train = pd.read_parquet(ROOT / 'data/nq_hotpotqa_train_30k_no_cold_start/train.parquet')
    old = pd.read_parquet(ROOT / 'data/nq_hotpotqa_train_30k_no_cold_start/validation_diagnostic_512.parquet')
    excluded = set(map(norm, train.question)) | set(map(norm, old.question))
    # Reserve the earlier study's diagnostic/evaluation/development panels.
    previous = ROOT / 'reports/evidence_gap_gpu23_20260915/panels.json'
    if previous.exists():
        for group in read(previous).values():
            for row in group:
                excluded.add(norm(row['question']))
    counters = collections.Counter(); candidates = []; seen = set()
    for row in test[test.data_source == 'hotpotqa'].to_dict('records'):
        counters['hotpot_total'] += 1
        q = row['question']; m = row['metadata']
        if norm(q) in excluded or norm(q) in seen:
            counters['excluded_overlap'] += 1; continue
        seen.add(norm(q))
        if m['type'] != 'bridge': continue
        counters['disjoint_bridge'] += 1
        titles = list(dict.fromkeys(m['supporting_facts']['title']))
        context = dict(zip(m['context']['title'], m['context']['sentences']))
        if len(titles) != 2 or not all(t in context for t in titles): continue
        counters['both_gold_docs_present'] += 1
        sf = collections.defaultdict(list)
        for t, i in zip(m['supporting_facts']['title'], m['supporting_facts']['sent_id']):
            if 0 <= i < len(context[t]): sf[t].append(context[t][i])
        choices = []
        for a, b in (titles, titles[::-1]):
            bridge = base_title(b)
            # Structural eligibility is independent of all model outputs.
            if not mention(base_title(a), q) or mention(bridge, q): continue
            if not mention(bridge, ' '.join(sf[a])): continue
            gold = list(row['golden_answers'])
            if not any(mention(g, ' '.join(sf[b])) for g in gold): continue
            if any(mention(g, ''.join(context[a])) for g in gold): continue
            d1, d2 = doc(a, context[a]), doc(b, context[b])
            if max(len(tok.encode(d['text'])) for d in (d1, d2)) > 512: continue
            distractions = [doc(t, ss) for t, ss in context.items() if t not in titles
                and not mention(bridge, t + ' ' + ''.join(ss))
                and not any(mention(g, t + ' ' + ''.join(ss)) for g in gold)]
            distractions = [d for d in distractions if len(tok.encode(d['text'])) <= 512]
            distractions.sort(key=lambda d: (abs(len(tok.encode(d['text'])) - len(tok.encode(d1['text']))), d['title']))
            if len(distractions) < 2: continue
            choices.append(dict(qid=str(row['id']), question=q, gold=gold,
                prompt=[dict(p) for p in row['prompt']], first=d1, second=d2,
                bridge=bridge, distractors=distractions[:2],
                first_support=list(sf[a]), second_support=list(sf[b])))
        if len(choices) == 1: candidates.append(choices[0])
    candidates.sort(key=lambda x: x['qid'])
    random.Random(2026091505).shuffle(candidates)
    counters['structurally_eligible'] = len(candidates)
    # Take the first 40 for input-only review; final inclusion is frozen before generation.
    panel = candidates[:40]
    if len(panel) < 32: raise RuntimeError(f'Only {len(panel)} eligible inputs')
    save(OUT / 'panel.json', panel)
    save(OUT / 'selection.json', dict(seed=2026091505, counters=dict(counters),
         panel_sha256=digest(panel), source=str(test_path), status='input review pending'))
    save(OUT / 'review.json', [dict(qid=x['qid'], include=None, reason='') for x in panel])
    lines = ['# Input-only review (no model outputs)\n']
    for x in panel:
        lines += [f"## {x['qid']}: {x['question']}", f"Answer: {x['gold']}",
                  f"First: {x['first']['title']} → second: {x['second']['title']}",
                  f"First support: {' '.join(x['first_support'])}",
                  f"Second support: {' '.join(x['second_support'])}\n"]
    (OUT / 'INPUT_REVIEW.md').write_text('\n\n'.join(lines))
    print(json.dumps(dict(counters), indent=2), flush=True)


def make_prefix(tok, q, condition):
    docs = [q['first']]
    if condition == 'evidence_removed': docs = [q['distractors'][0]]
    if condition == 'with_distractors':
        docs = q['distractors'].copy()
        docs.insert(int(hashlib.sha256(q['qid'].encode()).hexdigest(), 16) % 3, q['first'])
    if condition == 'both_hops': docs = [q['first'], q['second']]
    pre = tok.apply_chat_template(q['prompt'], tokenize=True, add_generation_prompt=True)
    history = '<think>I should look up information relevant to the question.</think>\n<search>' + q['first']['title'] + '</search>'
    observation = '\n'.join(f"Doc {i+1}(Title: {d['title']}) {d['text']}" for i, d in enumerate(docs))
    suffix = history + '\n\n<information>' + observation + '</information>\n\n'
    ids = pre + tok.encode(suffix, add_special_tokens=False)
    if len(ids) + 512 > 4096: raise RuntimeError('Context budget exceeded; no silent truncation')
    return ids, observation


def parse_action(text):
    m = re.search(r'<(search|answer)>(.*?)</\1>', text, re.S)
    return (m.group(1), m.group(2).strip(), text[:m.end()]) if m else ('invalid', None, text)


def generate_batch(model, tok, inputs):
    import torch
    from transformers import GenerationConfig, LogitsProcessor, StoppingCriteria
    class Restrict(LogitsProcessor):
        def __call__(self, input_ids, scores):
            scores[:, len(tok):] = -float('inf'); return scores
    class Stop(StoppingCriteria):
        def __init__(self, start): self.start = start
        def __call__(self, input_ids, scores, **kwargs):
            text = tok.batch_decode(input_ids[:, self.start:], skip_special_tokens=False)
            return torch.tensor(['</search>' in s or '</answer>' in s for s in text], device=input_ids.device)
    width = max(map(len, inputs))
    ids = torch.tensor([[tok.pad_token_id] * (width - len(x)) + x for x in inputs], device='cuda')
    mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in inputs], device='cuda')
    cfg = GenerationConfig(do_sample=False, max_new_tokens=512, use_cache=True,
        eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids('<|endoftext|>')],
        pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id)
    with torch.inference_mode():
        result = model.generate(ids, attention_mask=mask, generation_config=cfg,
            stopping_criteria=[Stop(width)], logits_processor=[Restrict()])
    decoded = tok.batch_decode(result[:, width:], skip_special_tokens=True)
    return [dict(raw_text=s, new_tokens=int((r != tok.pad_token_id).sum()))
            for s, r in zip(decoded, result[:, width:])]


class Retriever:
    def __init__(self):
        import requests
        self.session = requests.Session(); self.session.trust_env = False
        self.cache_path = OUT / 'retrieval_cache.json'
        self.cache = read(self.cache_path) if self.cache_path.exists() else {}

    def search(self, query):
        if query not in self.cache:
            r = self.session.post('http://127.0.0.1:8000/retrieve',
                json=dict(queries=[query], topk=3, return_scores=True), timeout=120)
            r.raise_for_status()
            self.cache[query] = r.json()['result'][0]
            save(self.cache_path, self.cache)
        return self.cache[query]


def summarize(results, checkpoints, panel):
    groups = collections.defaultdict(list)
    for r in results: groups[(r['label'], r['condition'])].append(r)
    summary = []
    for (label, condition), rows in groups.items():
        n = len(rows)
        summary.append(dict(label=label, condition=condition, n=n,
            second_hop_retrieval_success=sum(x['retrieval_success'] for x in rows)/n,
            next_query_bridge_mention=sum(x['bridge_in_query'] for x in rows)/n,
            immediate_answer_em=sum(x['answer_em'] for x in rows)/n,
            valid_search_rate=sum(x['action'] == 'search' for x in rows)/n,
            usable_next_action=sum(x['retrieval_success'] or x['answer_em'] for x in rows)/n,
            invalid_action_rate=sum(x['action'] == 'invalid' for x in rows)/n))
    save(OUT / 'summary.json', summary)
    import csv
    with (OUT / 'summary.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    lines = ['# Evidence-use GPU 5 pilot', '',
        'Existing checkpoints; exploratory matched-input evaluation, not independent training seeds.',
        f'Frozen panel: {len(panel)} HotpotQA bridge questions; {len(results)} completed continuations.',
        'The input-only review precedes generation. Exact document-title retrieval matches are conservative.',
        'Both-hops EM measures the first continuation only; requesting more search is recorded, not forcibly answered.',
        'No end-to-end accuracy or teacher-supervision improvement claim follows from this pilot.', '',
        '|Checkpoint|Condition|n|Second-hop retrieval@3|Immediate answer EM|Usable next action|',
        '|---|---|---:|---:|---:|---:|']
    for x in summary:
        lines.append(f"|{x['label']}|{x['condition']}|{x['n']}|{x['second_hop_retrieval_success']:.1%}|{x['immediate_answer_em']:.1%}|{x['usable_next_action']:.1%}|")
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU: raise RuntimeError('GPU 5 UUID binding required')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(20260915); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.device_count() == 1
    panel_all, review = read(OUT / 'panel.json'), read(OUT / 'review.json')
    assert digest(panel_all) == read(OUT / 'selection.json')['panel_sha256']
    assert len(review) == len(panel_all) and all(isinstance(r['include'], bool) and r['reason'] for r in review)
    selected = {r['qid'] for r in review if r['include']}
    assert {r['qid'] for r in review} == {q['qid'] for q in panel_all}
    panel = [q for q in panel_all if q['qid'] in selected]
    assert len(panel) >= 24
    checkpoints = [dict(label='initial-step0', method='initial', step=0, path='data/student/0.5B')]
    for step in (50, 100):
        for method, directory in [('opd', 'opd-grpo-0.5B'), ('er-alpha1.5', 'eropd-grpo-05B-test-alpha2')]:
            checkpoints.append(dict(label=f'{method}-step{step}', method=method, step=step,
                path=f'verl_checkpoints/{directory}/actor/global_step_{step}'))
    inputs_manifest = dict(checkpoints=checkpoints, panel_sha256=digest(panel_all),
        review_sha256=digest(review), selected_qids=[q['qid'] for q in panel],
        conditions=CONDITIONS, gpu_uuid=GPU, dtype='bfloat16', batch_size=8,
        max_new_tokens=512, seed=20260915, do_sample=False, retrieval_topk=3,
        note='ER directory says alpha2, but frozen training config is alpha=1.5; OPD and ER lambda=.01, both GRPO.')
    inputs_manifest['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    inputs_manifest['weight_files'] = {}
    for c in checkpoints:
        path = ROOT / c['path'] / 'model.safetensors'
        s = path.stat()
        inputs_manifest['weight_files'][str(path)] = dict(size=s.st_size, mtime_ns=s.st_mtime_ns)
    frozen = OUT / 'protocol.frozen.json'
    if frozen.exists(): assert read(frozen) == json.loads(json.dumps(inputs_manifest))
    else: save(frozen, inputs_manifest)
    tok = AutoTokenizer.from_pretrained(ROOT / 'data/student/0.5B', local_files_only=True)
    tok.pad_token = tok.eos_token; tok.padding_side = 'left'
    tasks = []
    for q in panel:
        for condition in CONDITIONS:
            ids, observation = make_prefix(tok, q, condition)
            tasks.append(dict(q=q, condition=condition, ids=ids, observation=observation))
    save(OUT / 'inputs.json', [dict(qid=t['q']['qid'], condition=t['condition'],
         prompt_ids=t['ids'], prompt_text=tok.decode(t['ids']), observation=t['observation']) for t in tasks])
    result_path = OUT / 'results.jsonl'
    results = [json.loads(s) for s in result_path.read_text().splitlines()] if result_path.exists() else []
    done = {(r['label'], r['qid'], r['condition']) for r in results}
    retriever = Retriever(); retriever.search('Albert Einstein')
    started = time.time()
    for ckpt in checkpoints:
        todo = [t for t in tasks if (ckpt['label'], t['q']['qid'], t['condition']) not in done]
        if not todo: continue
        path = ROOT / ckpt['path']
        assert (path / 'model.safetensors').exists(), path
        print('LOAD', ckpt['label'], str(path), flush=True)
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa', low_cpu_mem_usage=True).eval().to('cuda')
        for start in range(0, len(todo), 8):
            batch = todo[start:start+8]; tic = time.time()
            generated = generate_batch(model, tok, [t['ids'] for t in batch])
            for t, g in zip(batch, generated):
                q = t['q']; action, content, text = parse_action(g['raw_text'])
                docs = retriever.search(content) if action == 'search' and content else []
                titles = [d['document']['contents'].split('\n')[0].strip().strip('"') for d in docs]
                row = dict(label=ckpt['label'], step=ckpt['step'], qid=q['qid'],
                    condition=t['condition'], action=action, content=content, text=text,
                    raw_text=g['raw_text'], new_tokens=g['new_tokens'], input_tokens=len(t['ids']),
                    retrieval_titles=titles, target_title=q['second']['title'],
                    retrieval_success=any(norm(s) == norm(q['second']['title']) for s in titles),
                    bridge_in_query=action == 'search' and mention(q['bridge'], content or ''),
                    answer_em=exact(content if action == 'answer' else None, q['gold']))
                with result_path.open('a') as f: f.write(json.dumps(row, ensure_ascii=False) + '\n'); f.flush()
                results.append(row)
            summarize(results, checkpoints, panel)
            status = dict(state='running', checkpoint=ckpt['label'], completed=len(results),
                total=len(checkpoints)*len(tasks), last_batch_seconds=time.time()-tic,
                elapsed_seconds=time.time()-started, updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
            save(OUT / 'status.json', status); print('PROGRESS', json.dumps(status), flush=True)
        del model
        import gc
        gc.collect(); torch.cuda.empty_cache()
    save(OUT / 'status.json', dict(state='complete', completed=len(results),
         total=len(checkpoints)*len(tasks), elapsed_seconds=time.time()-started))
    print('COMPLETE', len(results), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'run'])
    args = p.parse_args()
    if args.stage == 'prepare': prepare()
    else:
        try: run()
        except Exception as e:
            save(OUT / 'failure.json', dict(error=repr(e), time=time.strftime('%Y-%m-%d %H:%M:%S')))
            raise
