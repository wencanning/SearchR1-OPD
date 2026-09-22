"""Independent CPU FP32 E5 + CPU FAISS retrieval. Never uses the training service."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--port', type=int, default=18085)
    p.add_argument('--witness-only', action='store_true')
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    import faiss
    import torch
    import datasets
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(4); torch.set_num_interop_threads(1); faiss.omp_set_num_threads(4)
    torch.manual_seed(20260916)
    start = time.monotonic()
    index_path = ROOT / 'data/retriever_root/wiki18/e5_Flat.index'
    corpus_path = ROOT / 'data/retriever_root/wiki18/wiki-18.jsonl'
    print('READ_CPU_INDEX', flush=True)
    index = faiss.read_index(str(index_path))
    print('READ_CORPUS', index.ntotal, index.d, flush=True)
    corpus = datasets.load_dataset('json', data_files=str(corpus_path), split='train', num_proc=4)
    assert len(corpus) == index.ntotal
    encoder_path = '/data/home/wencanning/.cache/huggingface/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd'
    tok = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)
    model = AutoModel.from_pretrained(encoder_path, local_files_only=True).float().eval().to('cpu')

    def retrieve(query):
        t = time.monotonic()
        x = tok(['query: ' + query], max_length=256, padding=True, truncation=True, return_tensors='pt')
        with torch.inference_mode():
            h = model(**x, return_dict=True).last_hidden_state
            h = h.masked_fill(~x['attention_mask'][..., None].bool(), 0.)
            embedding = torch.nn.functional.normalize(h.sum(1) / x['attention_mask'].sum(1)[..., None], dim=-1)
        assert torch.isfinite(embedding).all() and not torch.cuda.is_initialized()
        scores, ids = index.search(embedding.numpy().astype('float32'), 3)
        assert (ids >= 0).all()
        docs = [{'document': corpus[int(i)], 'score': float(s)} for i, s in zip(ids[0], scores[0])]
        with (args.out / 'queries.jsonl').open('a') as f:
            f.write(json.dumps(dict(query=query, indices=ids[0].tolist(), scores=scores[0].tolist(),
                seconds=time.monotonic()-t, cuda_initialized=False), ensure_ascii=False) + '\n')
        return docs

    witness = retrieve('Mikhail Lomonosov discovered what planet atmosphere')
    assert all(isinstance(d['document']['contents'], str) and d['document']['contents'] for d in witness)
    ready = dict(state='ready', pid=os.getpid(), ntotal=index.ntotal, dimension=index.d,
        cuda_initialized=torch.cuda.is_initialized(), encoder_dtype=str(next(model.parameters()).dtype),
        backend='CPU FP32 E5, CPU original FAISS index; differs from historical FP16 GPU index',
        witness_titles=[d['document']['contents'].split('\n')[0] for d in witness],
        seconds=time.monotonic()-start,
        source_metadata={str(q): dict(size=q.stat().st_size,mtime_ns=q.stat().st_mtime_ns) for q in (index_path,corpus_path)})
    (args.out / 'READY.json').write_text(json.dumps(ready, indent=2) + '\n')
    print('CPU_RETRIEVER_WITNESS_PASS', json.dumps(ready), flush=True)
    if args.witness_only:
        return
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    import threading
    lock = threading.Lock()
    app = FastAPI()
    class QueryRequest(BaseModel):
        queries: list[str]
        topk: int = 3
        return_scores: bool = True
    @app.get('/health')
    def health():
        return ready
    @app.post('/retrieve')
    def endpoint(request: QueryRequest):
        if len(request.queries) != 1 or request.topk != 3 or not request.return_scores:
            raise HTTPException(400, 'This diagnostic accepts one query, topk=3, return_scores=true only')
        with lock:
            return {'result': [retrieve(request.queries[0])]}
    uvicorn.run(app, host='127.0.0.1', port=args.port, workers=1, access_log=False)


if __name__ == '__main__':
    main()
