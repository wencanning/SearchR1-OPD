# GPU7 shared-capacity evaluation environment

Spec hash: f5285aa9. Existing conda reused, no installs. New resource shape and source instrumentation require a real witness before readiness.

- Working directory: /data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD
- Python: /data/home/wencanning/miniconda3/envs/searchr1/bin/python
- Physical GPU7 UUID GPU-c9944a46-db64-1215-948c-35aae45ebc32; launcher validates numeric mapping and exports CUDA_VISIBLE_DEVICES=7.
- Source: isolated frozen copy in research_runs/degradation_gpu7_20260920/source. Exact original agent/search/scoring protocol plus output recording, query cache, concurrency16.
- Shared retriever: http://127.0.0.1:8000/retrieve, existing GPU2/3 service.
- Owned GPU budget12GiB; no remaining-free-memory stop threshold. No other user's processes are stopped.
- Small actual inference witness; no training. Fresh agent must run invocation verbatim and write ENV_WITNESS.md. If it passes, write ENV_WITNESS_PASS.json including state=pass and SHA256 of launcher. Stop and report failure without fixing or retrying.

Invocation:
```bash
/data/home/wencanning/miniconda3/envs/searchr1/bin/python research_runs/degradation_gpu7_20260920/launch/run.py --checkpoint initial --panel witness_4 --name witness
```

Check reports/degradation_gpu7_20260920/witness/status.json and result.json: complete,4orderedtrain-only predictions,binaryscores,nooptimizerupdates,unchangedcheckpoint,peakowned<=12288MiB. Check trajectory includes real model generation/retrieval. An EM of0 is allowed. Record scorer disagreements rather than hiding them.

Validation status: PASS (fresh agent gpu7_eval_witness_0920, 2026-09-20). Four train-only cases completed, checkpoint hashes unchanged, no optimizer updates, sampled owned peak8274MiB. See reports/degradation_gpu7_20260920/ENV_WITNESS.md and ENV_WITNESS_PASS.json. Scope is an execution witness, not an accuracy claim or a universal memory guarantee.
