"""Manual-launch paired continuation diagnostic. Not a training algorithm.

Each state has identical native token prefixes, fixed checkpoint and budget.
No outcome-based selection; incomplete responses stay in the denominator.
Infrastructure failures abort instead of becoming zero rewards.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
GPU5 = 'GPU-2abc6681-2116-879c-a361-749edb32ac3b'
os.environ['HF_HUB_OFFLINE'] = '1'


class QueryCacheMiss(RuntimeError):
    """Infrastructure gap, never an observed task failure."""

    def __init__(self, query):
        self.query = query
        super().__init__('Query cache miss: provide --retriever-url, never invent evidence. Query: ' + repr(query))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def parse_forced(text, branch):
    match = re.fullmatch(r'\s*(.*?)</' + branch + r'>\s*', text, re.S)
    if not match or re.search(r'</?(?:search|answer|think|information)>', match[1]):
        return None
    return match[1].strip() or None


def trim_termination(tokens, eos_ids):
    """Environment observations continue the assistant trajectory, never after EOS."""
    end = next((i for i, token in enumerate(tokens) if token in eos_ids), len(tokens))
    return tokens[:end]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--states', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--retriever-url', help='Explicitly provided existing retrieval service; no service is started.')
    p.add_argument('--retrieval-cache', type=Path)
    p.add_argument('--retriever-manifest', type=Path, help='Optional frozen independent retriever provenance.')
    p.add_argument('--replicates', type=int, default=4)
    p.add_argument('--limit', type=int, default=0, help='Input-order pilot limit; 0=all, never select by outcomes.')
    p.add_argument('--witness-only', action='store_true', help='Seeded student forward/generation environment check; no retrieval.')
    args = p.parse_args()
    if args.replicates < 1 or args.limit < 0:
        p.error('replicates must be positive and limit nonnegative')
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif os.environ.get('CUDA_VISIBLE_DEVICES') != GPU5 or os.environ.get('ER_GPU_GUARDED') != '1':
        p.error('CUDA launch requires run_gpu5_guarded.py; physical GPU5 only.')
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, StoppingCriteria
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    args.checkpoint = args.checkpoint.resolve()
    cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    if cfg.hidden_size > 1024 or cfg.num_hidden_layers > 28:
        p.error('Only the 0.5B student is permitted by this launcher; no teacher loading.')
    if args.device == 'cuda':
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(6 * 1024**3 / total, 0)
    states_input = json.loads(args.states.read_text())
    for path, digest in states_input.get('source_sha256', {}).items():
        if sha(path) != digest:
            raise RuntimeError('Frozen state source changed: ' + path)
    states = states_input['states'][:args.limit or None]
    assert states and len({s['state_id'] for s in states}) == len(states)
    args.out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.out / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    weights = sorted(args.checkpoint.glob('*.safetensors')) + sorted(args.checkpoint.glob('pytorch_model*.bin'))
    assert weights, 'Checkpoint weights missing'
    helper = ROOT / 'scripts/experiments/evidence_use/experiment.py'
    manifest = dict(states_sha256=sha(args.states), code_sha256=sha(__file__), helper_sha256=sha(helper),
        checkpoint=str(args.checkpoint), checkpoint_config_sha256=sha(args.checkpoint / 'config.json'),
        weights={str(f): dict(size=f.stat().st_size, mtime_ns=f.stat().st_mtime_ns) for f in weights},
        state_ids=[s['state_id'] for s in states], replicates=args.replicates, device=args.device,
        dtype='float32' if args.device == 'cpu' else 'bfloat16',
        max_new_total=1024, max_segment=256, max_searches=3, max_context=4096,
        temperature=1.0, top_p=1.0, top_k=0, retriever_url=args.retriever_url,
        initial_cache_sha256=sha(args.retrieval_cache) if args.retrieval_cache else None,
        seed_base=2026091607, exploratory=states_input.get('exploratory', True))
    manifest['witness_only'] = args.witness_only
    manifest['termination_policy'] = 'strip first EOS/endoftext from continuation ids and text'
    manifest['retriever_manifest_sha256'] = sha(args.retriever_manifest) if args.retriever_manifest else None
    manifest['gpu_guard_sha256'] = sha(Path(__file__).with_name('run_gpu5_guarded.py')) if args.device == 'cuda' else None
    mf = args.out / 'execution.frozen.json'
    if mf.exists():
        assert json.loads(mf.read_text()) == manifest, 'Run manifest changed; use a new output directory.'
    else:
        save(mf, manifest)
    result_path = args.out / 'branches.jsonl'
    existing = [json.loads(l) for l in result_path.read_text().splitlines()] if result_path.exists() else []
    done = {(r['state_id'], r['branch'], r['replicate']) for r in existing}
    assert len(done) == len(existing), 'Duplicate branch keys'
    expected = {(s['state_id'], b, r) for s in states for b in ('answer','search') for r in range(args.replicates)}
    assert done <= expected, 'Unexpected branch keys in existing results'
    cache_path = args.out / 'retrieval_cache.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else (
        json.loads(args.retrieval_cache.read_text()) if args.retrieval_cache else {})
    import requests
    session = requests.Session(); session.trust_env = False
    def search(query):
        if query not in cache:
            if not args.retriever_url:
                raise QueryCacheMiss(query)
            r = session.post(args.retriever_url, json=dict(queries=[query], topk=3, return_scores=True), timeout=120)
            r.raise_for_status()
            docs = r.json()['result'][0]
            assert isinstance(docs, list)
            for d in docs:
                assert isinstance(d['document']['contents'], str)
            cache[query] = docs; save(cache_path, cache)
        return cache[query]
    tok = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
        torch_dtype=dtype, attn_implementation='eager', low_cpu_mem_usage=True).eval().to(args.device)
    sp = importlib.util.spec_from_file_location('fork_helpers', helper)
    M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M)
    class Vocab(LogitsProcessor):
        def __call__(self, ids, scores):
            scores[:, len(tok):] = -float('inf'); return scores
    eos_ids = [tok.eos_token_id, tok.convert_tokens_to_ids('<|endoftext|>')]
    def generate(ids, budget, forced=None):
        start = len(ids)
        class Stop(StoppingCriteria):
            def __call__(self, x, scores, **kw):
                text = tok.decode(x[0, start:], skip_special_tokens=False)
                yes = ('</' + forced + '>') in text if forced else '</answer>' in text or '</search>' in text
                return torch.tensor([yes], device=x.device)
        x = torch.tensor([ids], device=args.device)
        with torch.inference_mode():
            y = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=budget,
                do_sample=True, temperature=1., top_p=1., top_k=0, use_cache=True,
                eos_token_id=eos_ids, pad_token_id=tok.pad_token_id,
                stopping_criteria=[Stop()], logits_processor=[Vocab()])[0, start:].tolist()
        y = trim_termination(y, eos_ids)
        return y, tok.decode(y, skip_special_tokens=False)
    def preference(ids):
        x0 = torch.tensor([ids], device=args.device)
        with torch.inference_mode():
            h = model.model(x0, attention_mask=torch.ones_like(x0), use_cache=False).last_hidden_state[0, -1].float().cpu().tolist()
        values = []
        for name in ('answer', 'search'):
            tag = tok.encode('<' + name + '>', add_special_tokens=False)
            x = torch.tensor([ids + tag[:-1]], device=args.device)
            with torch.inference_mode():
                z = model(x, attention_mask=torch.ones_like(x), use_cache=False,
                    num_logits_to_keep=len(tag)).logits[0, :, :len(tok)].float().log_softmax(-1)
            values.append(float(z[torch.arange(len(tag), device=args.device), torch.tensor(tag, device=args.device)].sum()))
        return dict(logp_answer=values[0], logp_search=values[1], hidden_state=h,
            answer_probability=float(torch.sigmoid(torch.tensor(values[0] - values[1]))))
    if args.witness_only:
        torch.manual_seed(20260916)
        pref = preference(states[0]['prefix_ids'])
        tokens, decoded = generate(states[0]['prefix_ids'] + tok.encode('<answer>', add_special_tokens=False), 16, 'answer')
        assert tokens and all(__import__('math').isfinite(x) for x in pref['hidden_state'])
        result = dict(state='passed', device=args.device, hidden_width=len(pref['hidden_state']),
            tokens=tokens, decoded=decoded, cuda_initialized=torch.cuda.is_initialized(),
            allocated_peak_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0)
        save(args.out / 'WITNESS.json', result)
        print('GPU_FORK_WITNESS_PASS' if args.device == 'cuda' else 'CPU_FORK_WITNESS_PASS', result, flush=True)
        return
    for state in states:
        assert tok.decode(state['prefix_ids'], skip_special_tokens=False) == state['prefix_text'], 'Tokenizer changed native prefix'
        if all((state['state_id'], b, r) in done for b in ('answer', 'search') for r in range(args.replicates)):
            continue
        pref = preference(state['prefix_ids'])
        for rep in range(args.replicates):
            # Alternate execution order; independent branch seed, paired analysis by state.
            for branch in (('answer', 'search') if rep % 2 == 0 else ('search', 'answer')):
                key = (state['state_id'], branch, rep)
                if key in done:
                    continue
                seed = int(hashlib.sha256(f'2026091607|{key}'.encode()).hexdigest()[:8], 16)
                torch.manual_seed(seed)
                if args.device == 'cuda':
                    torch.cuda.manual_seed_all(seed)
                start = time.monotonic()
                ids = state['prefix_ids'] + tok.encode('<' + branch + '>', add_special_tokens=False)
                calls = used = 0; answer = None; actions = []; forced = branch
                while True:
                    budget = min(256, 1024 - used, 4096 - len(ids))
                    if budget <= 0:
                        stop = 'token_or_context_limit'; break
                    generated, text = generate(ids, budget, forced)
                    ids += generated; used += len(generated)
                    if forced:
                        content = parse_forced(text, forced); action = forced if content is not None else 'invalid'
                    else:
                        match = re.fullmatch(r'\s*<think>.*?</think>\s*<(search|answer)>(.*?)</\1>\s*', text, re.S)
                        action = match[1] if match else 'invalid'
                        content = match[2].strip() if match else None
                        if content is not None and re.search(r'</?(?:think|search|answer|information)>', content):
                            action = 'invalid'
                    forced = None
                    entry = dict(action=action, content=content, generated_ids=generated, raw_text=text)
                    actions.append(entry)
                    if action == 'answer':
                        answer = content; stop = 'answer_closed'; break
                    if action != 'search' or not content:
                        stop = 'invalid_or_incomplete'; break
                    if calls >= 3:
                        stop = 'search_limit'; break
                    try:
                        docs = search(content)
                    except QueryCacheMiss:
                        save(args.out / 'pending_retrieval.json', dict(
                            state_id=state['state_id'], qid=state['qid'], branch=branch,
                            replicate=rep, seed=seed, query=content, actions=actions,
                            search_calls=calls, generated_tokens=used,
                            seconds=time.monotonic() - start,
                            cuda_initialized=torch.cuda.is_initialized(),
                            task_outcome_observed=False))
                        save(args.out / 'status.json', dict(state='blocked_cache_miss',
                            completed=len(done), total=len(expected)))
                        raise
                    calls += 1
                    body = '\n'.join(f"Doc {i+1}(Title: {d['document']['contents'].split(chr(10))[0].strip()}) " +
                        '\n'.join(d['document']['contents'].split('\n')[1:]) for i, d in enumerate(docs))
                    body_ids = tok.encode(body, add_special_tokens=False)
                    obs = tok.encode('\n\n<information>', add_special_tokens=False) + body_ids[:512] + tok.encode('</information>\n\n', add_special_tokens=False)
                    ids += obs
                    entry.update(query=content, observation_ids=obs, observation_truncated=len(body_ids) > 512)
                row = dict(state_id=state['state_id'], qid=state['qid'], branch=branch, replicate=rep,
                    seed=seed, preference=pref, final_answer=answer, em=M.exact(answer, state['gold']),
                    search_calls=calls, generated_tokens=used, stop_reason=stop, actions=actions,
                    seconds=time.monotonic() - start, device=args.device,
                    cuda_initialized=torch.cuda.is_initialized(),
                    cuda_max_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else 0,
                    cuda_max_reserved_bytes=torch.cuda.max_memory_reserved() if args.device == 'cuda' else 0)
                with result_path.open('a') as f:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n'); f.flush(); os.fsync(f.fileno())
                done.add(key)
                save(args.out / 'status.json', dict(state='running', completed=len(done), total=len(states)*2*args.replicates))
                print('BRANCH', key, row['em'], stop, flush=True)
    assert len(done) == len(states)*2*args.replicates
    for f in weights:
        assert manifest['weights'][str(f)] == dict(size=f.stat().st_size, mtime_ns=f.stat().st_mtime_ns)
    save(args.out / 'status.json', dict(state='complete', completed=len(done), total=len(done)))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if '--out' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--out') + 1])
            if output.is_dir():
                save(output / 'failure.json', dict(state='failed', error=repr(error)))
        raise
