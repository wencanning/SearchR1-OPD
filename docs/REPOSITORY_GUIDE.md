# 目录与运行说明

## 代码与实验

| 目录 / 文件 | 用途 |
| --- | --- |
| `verl/` | PPO、OPD、SOD、ER-OPD 训练器与 rollout 实现 |
| `search_r1/` | 搜索交互、生成逻辑和检索服务 |
| `train_*.sh` | 训练入口；运行前检查模型路径、GPU、实验名及超参数 |
| `scripts/nq_hotpotqa/` | 数据准备与完整评测入口 |
| `scripts/baseline/` | 基线、续训和任务接续脚本 |
| `scripts/diagnostics/` | 信用分配、证据、历史及搜索行为诊断 |
| `scripts/experiments/` | 按研究问题或日期归档的实验实现 |
| `tests/` | 算法、数据和生成协议测试 |
| `results/` | 可提交的精简结果、来源和配置 |
| `figures/` | 图与源数据；名称含 SYNTHETIC / mockup 的文件仅为示意，不是实验结果 |
| `docs/` | 方法说明、实验设计和论文草稿 |
| `.aris/compute/` | 历史环境规格与验证记录，含原机器路径，仅用于复现参考 |

## 运行环境

安装说明见根目录 VERL_README.md。现有实验使用 conda `searchr1`；部分 CPU 诊断使用 `.venv`，二者不要混用。日期归档脚本可能含原机器路径，需要先检查配置；其依赖的数据、checkpoint、reports 不随 Git 发布。

`train_er_opd.sh` 当前保存的是 1.5B（本地目录名 1B）、alpha0.5 的训练设置，并非所有实验共用的默认协议。启动前检查脚本内容。`scripts/nq_hotpotqa/evaluate.sh` 要求设置 BASE_MODEL，可用 PYTHON_BIN 指定解释器，通过末尾 Hydra 参数覆盖 batch size 等配置。

## 本地产物

以下目录保留在工作区，但不上传：`data/`、`verl_checkpoints/`、`outputs/`、`wandb/`、`*logs/`、`reports/`、`research_runs/`（含重复源码快照及任务状态）、本地自动生成的 `MANIFEST.md`、`.aris` 的 traces/meta/external 和锁文件。没有删除这些实验数据。

需要分享的新结果应整理到 `results/<experiment>/`，至少保留模型身份、checkpoint step、评测参数、指标和原始结果校验值。仅目录名不能证明实际 alpha；结果身份冲突和不同训练步数应明确标注。

## 验证

在具备相应依赖的环境中执行：

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

部分历史测试/诊断可能需要另一套环境或本地资产。提交前的实际检查及限制记录在本次提交说明中；完整训练和评测依赖本地模型及检索数据。

本次整理验证（2026-09-22）：在现有 conda `searchr1` 环境中以 CPU 模式运行上述 unittest 命令，110 项测试全部通过；新增/修改 Python 与 shell 文件语法、结果来源 SHA-256、README 导航链接及 Git 差异检查通过。未重跑完整训练；五组评测结果来自已完成的 GPU 2/3 队列。
