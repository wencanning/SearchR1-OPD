## 实验安排

### 主实验

- 在写论文时要声明，蒸馏方法（SOD、OPD）默认加入 GRPO 来增加稳定性。
- Single-Hop：NQ、TriviaQA、PopQA；Multi-Hop：HotpotQA、2wiki、Musique、Bamboogle。

| Params | Method | NQ | TriviaQA | PopQA | HotpotQA | 2wiki | Musique | Bamboogle | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7B | GRPO | 0.429 | 0.623 | 0.427 | 0.386 | 0.346 | 0.162 | 0.4 | 0.396 |
| 0.5B | Vanilla | | | | | | | | |
| | SFT | 0.319 | 0.471 | 0.371 | 0.277 | 0.289 | 0.094 | 0.184 | 0.287 |
| | GRPO | 0.323 | 0.476 | 0.371 | 0.267 | 0.242 | 0.087 | 0.193 | 0.279 |
| | OPD | 0.306 | 0.457 | 0.369 | 0.252 | 0.227 | 0.065 | 0.153 | 0.261 |
| | SOD | | | | | | | | |
| | **OUR** | **0.339** | **0.495** | **0.413** | **0.289** | **0.277** | **0.082** | **0.192** | **0.298** |
| 1.5B | Vanilla | | | | | | | | |
| | SFT | 0.379 | 0.571 | 0.393 | 0.358 | 0.342 | 0.145 | 0.296 | 0.355 |
| | GRPO | 0.402 | 0.561 | 0.428 | 0.351 | 0.304 | 0.123 | 0.282 | 0.350 |
| | OPD | 0.373 | 0.551 | 0.387 | 0.313 | 0.283 | 0.106 | 0.315 | 0.333 |
| | SOD | | | | | | | | |
| | **OUR** | | | | | | | | |

SFT 数值来自各自 step 375 checkpoint 的完整评测，保留三位小数：

- 0.5B：[原始评测日志](eval_loop_logs/sft_comparison_20260911_223945/SFT_0.5B.log)。
- 1.5B：[修复后评测日志](eval_loop_logs/sft_15b_fixed_metadata_20260912/eval.log)。
- 注意：0.5B 使用评测设置传递修复前的代码，1.5B 使用修复后的代码；该修复涉及多轮生成中的采样设置及 log-prob 重算开关传递。正式比较前需统一评测实现。

### 消融实验

- Pure ER-OPD：不加入 GRPO。
- 不冷启动。
- $\alpha \in \{0, 0.5, 1, 1.5, 2\}$。
  - 目前的成绩对应 $\alpha = 1$；$\alpha = 0$ 为普通 OPD。

## 改进

- 将 $2z_{\mathrm{obs}} - z_{\mathrm{hid}}$ 改为带 $\alpha$ 的一般形式：

$$
z_{\mathrm{ER}} = z_{\mathrm{obs}} + \alpha\left(z_{\mathrm{obs}} - z_{\mathrm{hid}}\right)
= (1 + \alpha)z_{\mathrm{obs}} - \alpha z_{\mathrm{hid}}.
$$
