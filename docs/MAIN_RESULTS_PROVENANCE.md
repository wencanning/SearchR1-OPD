# 主表结果来源

0.5B 的两行按指定展示对应关系填写：OPD 行使用 alpha 0.5 的结果，ER-OPD 行使用 alpha 0 的结果。实际 checkpoint 来源如下；其中 alpha 0 是普通 observed OPD checkpoint，这里的展示标签不改变其训练方法归属。两组均为 step 150，GPU 2、3，val batch size 512，seed 42，完整评测 102/102 个 batch，分数保留三位小数。

- OPD 行：`eropd-alpha05-from-6bpwblvv-step50-20260912_164223/actor/global_step_150`（alpha 0.5），[评测结果](../results/eropd_alpha_sweep_20260921/alpha0.5.json)。
- ER-OPD 行：`opd-grpo-0.5B/actor/global_step_150`（alpha 0），[评测结果](../results/eropd_alpha_sweep_20260921/alpha0.json)。

Vanilla 使用官方 Qwen2.5-Instruct 0.5B / 1.5B 权重，未经本项目的 SFT 或后续训练（模型本身已接受官方指令微调）。两次完整评测均完成 203/203 个 batch；分数保留三位小数，Avg. 为七个数据集分数的算术平均。

- 0.5B：[评测结果](../results/baselines/vanilla_05b.json)、原始评测日志（本地保留：`eval_loop_logs/vanilla_instruct_students_gpu2_20260917_192712_79820/0.5B/eval.log`）。
- 1.5B：[评测结果](../results/baselines/vanilla_15b.json)、原始评测日志（本地保留：`eval_loop_logs/vanilla_instruct_students_gpu2_20260917_192712_79820/1.5B/eval.log`）。

SFT 数值来自各自 step 375 checkpoint 的完整评测，保留三位小数：

- 0.5B：原始评测日志（本地保留：`eval_loop_logs/sft_comparison_20260911_223945/SFT_0.5B.log`）。
- 1.5B：修复后评测日志（本地保留：`eval_loop_logs/sft_15b_fixed_metadata_20260912/eval.log`）。
- 注意：0.5B 使用评测设置传递修复前的代码，1.5B 使用修复后的代码；该修复涉及多轮生成中的采样设置及 log-prob 重算开关传递。正式比较前需统一评测实现。

SOD 0.5B 数值来自 `sod-grpo-0.5B/actor/global_step_150` 的完整评测（203/203 个 batch），保留三位小数，Avg. 为七个数据集分数的算术平均。来源：[评测结果](../results/baselines/sod_05b.json)、原始评测日志（本地保留：`eval_loop_logs/sod_05b_step150_gpu2_20260917_133247_3011937/eval.log`）。

SOD 1.5B 数值来自 `sod-grpo-coef0.5-1B/actor/global_step_150` 的完整评测，保留三位小数，Avg. 为七个数据集分数的算术平均。checkpoint 目录中的 `1B` 对应实际的 1.5B 模型；本次使用评测设置传递修复后的代码。来源：[评测结果](../results/baselines/sod_15b.json)、原始评测日志（本地保留：`eval_loop_logs/sod_15b_after_sft_fixed_20260912/eval.log`）。
