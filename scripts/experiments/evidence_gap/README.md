# Evidence helps, teacher still wrong：GPU 2/3 实验

这套脚本把论文草图转换成真实数据实验。**现在没有启动任何 GPU 实验，也没有修改正在运行的训练。** 建议先跑 pilot，确认现象和资源开销，再跑正式三方法比较。

## 先运行这个

在仓库根目录：

```bash
bash scripts/experiments/evidence_gap/run_gpu23.sh pilot
```

固定物理 **GPU 2、3**，按 UUID 映射，两个独立 HF worker 每卡处理一半问题。若两卡任一卡存在计算进程或显存占用超过 1 GiB，启动器直接退出，打印占用 PID；请等现有任务释放后再跑。不会自动结束旧训练，不执行 `ray stop`。

pilot 使用 **32 道固定 HotpotQA 问题 × 学生检查点 0、75、150**，生成自然检索轨迹，收集普通/隐藏证据教师的答案，输出盲审材料。旧检查点来自 OPD 无 GRPO 消融，所以它仅用于初步现象诊断，不能充当正式方法比较。教师固定为 SearchR1 7B，学生为 0.5B。

脚本前台运行；要离开终端，请在你自己的 tmux 会话里运行。日志在：

```text
reports/evidence_gap_gpu23_20260915/pilot/collect-gpu2.log
reports/evidence_gap_gpu23_20260915/pilot/collect-gpu3.log
```

这一阶段在生成审查文件后正常结束。**不是故障，也不会偷跑正式九组训练。** 可先把这些文件交给我继续检查：

```text
reports/evidence_gap_gpu23_20260915/pilot/review_cards.json
reports/evidence_gap_gpu23_20260915/pilot/annotations.template.json
```

## 实验回答什么

|图|横轴|纵轴与含义|
|---|---|---|
|(a)|Student training step|证据充分且可计分的状态中，教师已偏向正确 / 证据有帮助但仍偏错 / 没有有益更新，各占多少|
|(b)|Student training step|在普通教师定义的“有帮助但仍偏错”子集中，hidden、observed、ER、熵匹配教师对正确候选的相对支持；是有限候选支持度，不是准确率|
|(c)|Student training step|三种匹配训练方法在同一留出集初始错误子集上的实际生成纠错率，按参考答案 EM 计算；同时输出原本正确的保持率和整体 EM/F1|

完整候选分数用别名类别的概率质量，包含 `</answer>` 终止字符串；不会只比较第一个 token。每个别名的逐token分数都会保存。B 除了图上的支持度，还输出自由生成、候选翻转以及全体充分证据状态的净纠错/净伤害，避免只展示挑选出来的正例。

## 数据与预算已固定

- 训练：当前 parquet 里仅 **20,000 条 HotpotQA**；10,000 条 NQ 不参与。
- 诊断 200 题、最终评估 512 题、开发验证 128 题，来自 HotpotQA 留出数据，三个面板互斥，并排除训练题及旧 diagnostic validation 中出现的问题。
- 正式检查点：**0、25、50、75、100、125、150**。缺检查点就报错，不用附近步数代替。
- 正式训练：**OPD+GRPO、SOD+GRPO、ER-OPD+GRPO × seeds 42/43/44**，每组150次实际更新。两GPU共同训练一组，九组顺序执行。
- batch128，每题8条rollout，GRPO系数1，distillation λ=.01，lr1e-6。ER主实验 α=1；不覆盖你当前 α=1.5 的运行。
- 诊断教师 FP32，训练教师 FP16；诊断不是对历史 FP16 logits 的严格复现。相同步数/rollout预算不等于相同计算量。
- 所有值在 `protocol.json`。已经冻结面板后改配置应另设 `output` 路径；原目录拒绝混入不同协议。

CPU 已准备好的 `panels.json` 只含数据输入，**不是 GPU 实验结果**。正式任务小时数尚未实测，先用 pilot/单步 smoke 的时间测量估算，不虚报耗时。

## 完整执行顺序

### 1. 只检查或只查看命令

```bash
bash scripts/experiments/evidence_gap/run_gpu23.sh check
bash scripts/experiments/evidence_gap/run_gpu23.sh train --seeds 42 --dry-run
```

`check` 会因为当前GPU占用而返回非零。`--dry-run` 不启动模型、检索或训练。已有检索服务必须在 `http://127.0.0.1:8000/retrieve` 可用；脚本不会擅自另开服务。

### 2. 小规模诊断与候选审查

```bash
bash scripts/experiments/evidence_gap/run_gpu23.sh pilot
cp reports/evidence_gap_gpu23_20260915/pilot/annotations.template.json \
   reports/evidence_gap_gpu23_20260915/pilot/annotations.json
```

逐条查看 `review_cards.json`，编辑 `annotations.json`：

```json
{
  "review_id": "R00001",
  "key": "保留模板中的原值",
  "reviewed": true,
  "reviewer": "审查者姓名或编号",
  "sufficient": true,
  "evidence_reason": "指出实际可见检索文本如何覆盖问题需要的两跳事实",
  "correct_aliases": ["经过审查的正确答案", "合法别名"],
  "wrong_classes": [["一个错误对象的名称", "该错误对象的别名"], ["另一个错误对象"]],
  "notes": "参考歧义、分歧协调等备注"
}
```

这是格式示例，不能原样填入。充分性不能靠字符串命中自动确定。错误类别先从学生、普通/hidden教师的错误续写归并；这些答案也可能是合法别名，需要审查。学生已经答对的状态同样保留，需给出有依据的错误对照候选。证据不充分或技术排除的行也要保留并完成复核；不充分行无需错误类别。

建议两位审查者独立判断并协调分歧。模板记录最终判断，不假装自动完成了双人审查。ER分数/生成不会出现在这一轮材料中。随后：

```bash
bash scripts/experiments/evidence_gap/run_gpu23.sh score --label pilot
```

所有审查完成才能开始打分。第一次 score 会冻结标签；若需修订，另建带明确修订说明的实验目录，不事后覆盖主分析。输出 `A.csv`、`B.csv`、`denominators.csv`、`generation.csv`、`summary.json` 和新的生成语义审查文件。非别名 EM 命中的生成仍可能语义正确，所以输出只能称“reference/alias EM”，不能当作人工语义准确率。

### 3. GPU 单步训练验证

```bash
bash scripts/experiments/evidence_gap/run_gpu23.sh smoke --seeds 42
```

三方法各跑 **1次更新**，路径在 `.../smoke/`。用于验证真实 FSDP/vLLM/显存条件，不进入论文。CPU配置解析通过不等于GPU训练通过。

### 4. 正式三方法训练

```bash
# 先做同一个seed的三方法，检查资源与训练行为
bash scripts/experiments/evidence_gap/run_gpu23.sh train --seeds 42

# 补齐其余seed；完整组自动跳过
bash scripts/experiments/evidence_gap/run_gpu23.sh train --seeds 43 44
```

三种方法从同一隔离源码副本运行；不修改现有训练入口。每组命令与日志保存到 `.../training/{opd,sod,er}-seed{42,43,44}/`。不自动挑最优检查点，不自动改变微批量/方法参数。失败的训练不会被伪装成优化器续训；脚本保留日志并退出，需要明确的新输出目录重启实验。诊断/计分可按已完成记录续跑。

### 5. 正式 A/B

```bash
bash scripts/experiments/evidence_gap/run_gpu23.sh collect --method opd --seeds 42
```

输出目录 `.../diagnostic-opd-seed42/`；200题×7步。按第2步同样流程完成该目录的审查后：

```bash
bash scripts/experiments/evidence_gap/run_gpu23.sh score --label diagnostic-opd-seed42
```

正式A固定参考为OPD seed42，不能挑最有利seed或混合ER学生轨迹来制造现象。其他seed可作为独立复核，但应保留单独目录与图注。

### 6. 正式 C 与绘图

```bash
bash scripts/experiments/evidence_gap/run_gpu23.sh evaluate
bash scripts/experiments/evidence_gap/run_gpu23.sh summarize --label diagnostic-opd-seed42
uv run --with matplotlib==3.11.2 python scripts/experiments/evidence_gap/plot.py \
  reports/evidence_gap_gpu23_20260915/diagnostic-opd-seed42/summary.json
```

绘图用独立 uv 环境，不改训练环境。全英文、折线与置信区间，输出 PNG/PDF/SVG。C只在全部九组、全部步数、全部512题完成后输出，否则只画真实AB，不填占位分数。A同时输出每步完整回答、无回答、边界无效、缺证据和审查排除计数；B分别输出新增纠错与新增错误，不能用净值抵消伤害。还会检查各运行共同step0生成是否一致；若不一致，先排查检索库/eval漂移，不能直接合并。

最终产物：`.../diagnostic-opd-seed42/evidence_gap_REAL.pdf`。脚本完全不读取之前的合成草图数据。

## 技术边界与验证记录

新诊断脚本在工具观察截断时保留与标签共用的边界 token，让协议完整；该token内少量正文也会保持可见。新训练也使用同一修正规则：启动器将当前 `verl/` 和 `search_r1/` 复制到实验目录的 `training_source/`，仅在这个隔离副本中修正边界，并从该副本启动三种方法。`OBSERVATION_BOUNDARY.patch` 记录唯一源码补丁。原工作区和现有训练不变。HF诊断仍不宣称完全复现vLLM采样轨迹。教师仍可能通过学生历史访问已转述的信息，hidden干预只移除直接证据key访问。

当前CPU测试覆盖同预算配置、标签冻结、分组、熵匹配、微型Qwen模型缓存/无缓存隐藏证据一致性、检索标签保留、完整搜索轨迹与原生答案边界、统计分母。三方法Hydra完整配置已使用现有训练Python解析。GPU真实加载/单步训练验证仍待用户运行。

详细实验判据、统计口径、负结果解释：[实验计划](../../../refine-logs/evidence_gap_gpu23_20260915/EXPERIMENT_PLAN.md)。
