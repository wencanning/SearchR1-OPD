# ER后续实验：先测有用性，再决定方法

本目录是下一阶段的**诊断采集与分析脚本**，不是已实现的新方法训练器。大型实验尚未启动。

已完成：CPU逐题偏置分析；64/64自然题目的检索后原生token状态冻结；真实0.5B CPU特征/规范动作概率前向检查。未完成：完整真实检索分支端到端运行、GPU运行检查、大规模训练。

## 用户手动运行

在仓库根目录，使用已存在的检索服务地址。以下8000地址与历史实验一致，脚本不启动或重启检索服务。**检索服务的设备资源由该服务自身管理，GPU5限制只覆盖新启动的学生推理进程。** 如果该服务当前承载忙碌的训练检索，请安排合适时段。

先做4分支计时pilot：

```bash
bash scripts/experiments/er_next/run_manual_gpu5.sh pilot http://127.0.0.1:8000/retrieve
```

确认pilot完成且输出合理，再由你手动启动512分支探索性采集：

```bash
bash scripts/experiments/er_next/run_manual_gpu5.sh full http://127.0.0.1:8000/retrieve
```

GPU5学生allocator限制6GiB；启动至少20GiB空闲；每秒检查自身进程，达到8GiB或全卡剩余低于12GiB即停止**自己**。不加载7B教师，不使用其他卡，不停止别人的进程。监控有采样延迟，不能承诺对其他进程突增无影响；本轮自主工作全部使用CPU。

CPU运行同一采集器（可能很慢，仍由用户启动）：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
nice -n 10 .venv/bin/python scripts/experiments/er_next/collect_forks.py \
  --states reports/er_next_20260916/fork_states.frozen.json \
  --checkpoint verl_checkpoints/opd-grpo-0.5B/actor/global_step_50 \
  --out reports/er_next_20260916/forks_cpu_pilot \
  --device cpu --limit 2 --replicates 1 \
  --retrieval-cache reports/evidence_stopping_gpu5_20260915/retrieval_cache.json \
  --retriever-url http://127.0.0.1:8000/retrieve
```

## 输出与边界

- `execution.frozen.json`：代码、输入、checkpoint配置hash和权重metadata；不是全权重加密hash。
- `branches.jsonl`：每个状态每个动作每次续写一条，包含query、观察token、终止原因、EM、时间和动作前特征。
- `retrieval_cache.json`：真实查询的检索结果；cache miss且无服务地址时失败，不伪造证据。
- `status.json`：必须complete且满足所有分支key才能分析；重启同一命令会校验manifest并跳过已完成分支。半写JSONL须人工检查，不能自动删除以掩盖故障。
- `FORK_REPORT.md`、`fork_summary.json`：探索性收益分析。4次续写前半用于选动作/拟合，后半用于评估；不是已训练策略的最终准确率。

64题均为先前看过的探索性面板。不得将其拿来训练或充当全新确认集。大训练必须等待全新问题确认性结果和完整训练脚本；没有后台自动升级到训练的行为。

严格正文和纯格式补充两臂已完成；严格正文292token全部是查询，不能解释为答案正文训练。4题36生成pilot未观察到方法间EM或动作差异。共享参数导致正文训练也会影响动作，所以“mask动作”不意味着动作概率保持不变。

## 最新CPU结果

`reports/er_next_20260916/forks_cpu_cache_pilot`：不提供retriever URL的2状态4分支小试验，只完成1个回答终局；搜索出现缓存外新查询后退出。`pending_retrieval.json`保留缺失查询与分支，`status.json`为blocked_cache_miss。缺失检索不计EM=0，当前不能估计两支收益。新的GPUpilot独立存储，不能与CPU半成品拼接。

`reports/retrieval_redundancy_cpu_20260916/REPORT.md`：旧320条自然轨迹576次检索已全部审计。step50无新文档调用23→10，step100为4→16；两个step合计少22次搜索，其中无新文档调用只净少1次。不支持“ER稳定减少冗余”的解释。文档新颖性也不等于搜索收益，下一步仍需真实分支续写。
