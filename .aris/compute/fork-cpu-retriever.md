# Independent CPU retriever environment: pending witness

Working directory: /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD.
Use existing retriever conda Python, no install or rebuild. Spec: .aris/compute/fork-cpu-retriever-spec.json.
4 CPU threads, estimated <90GiB RAM (61GiB index plus encoder/corpus mappings); CUDA hidden. No HTTP requests and no existing process changes. Witness loads the full original index, encodes one real query and fetches actual corpus records; CUDA must remain uninitialized. This is a CPU retrieval backend, not precision-identical to the old GPU FP16 index.

Run verbatim:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 nice -n 10 /data/home/wencanning/miniconda3/envs/retriever/bin/python scripts/experiments/er_next/cpu_retriever.py --out reports/er_next_20260916/cpu_retriever_witness --witness-only
```

Expected: exit0, CPU_RETRIEVER_WITNESS_PASS, READY.json with ntotal matching corpus size, three real documents, CUDA false. Do not improvise fixes or install packages; report discrepancies. The full index may take minutes to load. Check progress at <=60s intervals.
