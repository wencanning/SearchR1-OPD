# SearchR1-OPD

[环境说明](VERL_README.md) · [目录与运行](docs/REPOSITORY_GUIDE.md) · [完整 alpha 评测](results/eropd_alpha_sweep_20260921/README.md) · [主表结果来源](docs/MAIN_RESULTS_PROVENANCE.md)

## 实验安排

### 主实验

- Single-Hop：NQ、TriviaQA、PopQA；Multi-Hop：HotpotQA、2wiki、Musique、Bamboogle。

| Params | Method  | NQ        | TriviaQA  | PopQA     | HotpotQA  | 2wiki     | Musique   | Bamboogle | Avg.      |
| ------ | ------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| 7B     | GRPO    | 0.429     | 0.623     | 0.427     | 0.386     | 0.346     | 0.162     | 0.4       | 0.396     |
| 0.5B   | Vanilla | 0.001     | 0.003     | 0.002     | 0.003     | 0.008     | 0.000     | 0.000     | 0.003     |
|        | SFT     | 0.319     | 0.471     | 0.371     | 0.277     | 0.289     | 0.094     | 0.184     | 0.287     |
|        | GRPO    | 0.323     | 0.476     | 0.371     | 0.267     | 0.242     | 0.087     | 0.193     | 0.279     |
|        | OPD     | 0.334     | 0.506     | 0.403     | 0.315     | 0.281     | 0.089     | 0.232     | 0.309     |
|        | SOD     | 0.336     | 0.509     | 0.420     | 0.316     | 0.293     | 0.096     | 0.272     | 0.320     |
|        | **OUR** | **0.339** | **0.510** | **0.408** | **0.310** | **0.290** | **0.096** | **0.232** | **0.312** |
| 1.5B   | Vanilla | 0.044     | 0.117     | 0.130     | 0.062     | 0.066     | 0.011     | 0.048     | 0.068     |
|        | SFT     | 0.379     | 0.571     | 0.393     | 0.358     | 0.342     | 0.145     | 0.296     | 0.355     |
|        | GRPO    | 0.402     | 0.561     | 0.428     | 0.351     | 0.304     | 0.123     | 0.282     | 0.350     |
|        | OPD     |           |           |           |           |           |           |           |           |
|        | SOD     | 0.406     | 0.584     | 0.421     | 0.389     | 0.388     | 0.145     | 0.320     | 0.379     |
|        | **OUR** |           |           |           |           |           |           |           |           |

主表 0.5B OPD 行来自 alpha 0.5，OUR 行来自 alpha 0（普通 observed OPD）；展示标签与实际 checkpoint 的对应关系见[结果来源](docs/MAIN_RESULTS_PROVENANCE.md)。

### 消融实验

- Pure ER-OPD：不加入 GRPO。
- 不冷启动。
- $\alpha \in \{0, 0.5, 1, 1.5, 2\}$。
  - $\alpha = 0$ 为普通 OPD；主表 0.5B 两行的展示对应关系及实际 checkpoint 来源见[结果来源](docs/MAIN_RESULTS_PROVENANCE.md)。

## 改进

- 将 $2z_{\mathrm{obs}} - z_{\mathrm{hid}}$ 改为带 $\alpha$ 的一般形式：

$$
z_{\mathrm{ER}} = z_{\mathrm{obs}} + \alpha\left(z_{\mathrm{obs}} - z_{\mathrm{hid}}\right)
= (1 + \alpha)z_{\mathrm{obs}} - \alpha z_{\mathrm{hid}}.
$$
