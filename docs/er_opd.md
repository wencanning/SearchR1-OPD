# Evidence-Residual On-Policy Distillation（ER-OPD）

## 1. 研究动机

### 1.1 问题背景

Search-R1 类模型需要在一条轨迹中交替完成：

1. 内部推理；
2. 发起搜索动作；
3. 读取检索结果；
4. 基于证据继续推理并给出答案。

传统的 On-Policy Distillation（OPD）让教师模型在学生实际采样的轨迹上
打分，并直接把教师的 token 分布作为蒸馏目标。该方式能够把教师的语言建模
能力传给学生，但普通 OPD 只回答了“教师在当前完整上下文下偏好什么”，没有
显式区分：

- 当前预测是由检索证据支持的；还是
- 当前预测主要来自教师自身的先验或与证据无关的上下文。

对于搜索增强推理，这一区分很重要。我们希望学生学习的是“如何利用检索到的
证据进行后续推理”，而不仅是模仿一个在完整轨迹上更强的语言模型。

### 1.2 核心想法

ER-OPD 将教师在完整证据上下文中的输出，与教师在屏蔽检索正文后的输出进行
对比，把两者的差异视为证据带来的 residual（残差）：

> 教师在看到证据后，相对于看不到证据时改变了哪些 token 偏好？

随后放大这部分证据相关变化，并将放大后的分布作为 OPD 教师目标。这样，蒸馏
信号不仅包含教师的原始预测，还显式包含证据对教师预测的增量影响。

### 1.3 设计目标

ER-OPD 的实现围绕以下目标展开：

- **证据敏感**：教师目标应反映检索证据带来的预测变化；
- **轨迹一致**：教师评价学生实际采样的 reasoning/search/answer 轨迹，而不是
  额外构造一条离线教师轨迹；
- **动作可训练**：检索结果 `<information>...</information>` 的正文不作为
  学生的策略输出，不对其计算策略损失或蒸馏优势；
- **目标可审计**：保留 observed、hidden 和 ER 三种分布的熵、log-probability
  及 residual 统计量；
- **与现有 OPD 兼容**：只替换教师 token log-probability 的计算方式，继续复用
  现有的 K1 reverse-KL、GRPO outcome reward 和 PPO actor update。

## 2. 具体的方法

### 2.1 轨迹与证据标注

学生使用标准 Search-R1 rollout 生成多轮轨迹。每一轮可以包含 assistant 的
推理/搜索动作，以及环境返回的检索观察结果。

对环境返回的 observation 做 tokenization 时，代码识别
`<information>...</information>` 区间，并构造与输入序列对齐的
`evidence_mask`：

- `<information>` 和 `</information>` 边界 token 保持可见；
- 只有边界内部的检索正文 token 被标记为 evidence；
- padding 不允许被标记为 evidence；
- observation token 进入上下文，但不进入学生的策略损失和 OPD 蒸馏损失。

因此，ER-OPD 使用的是“同一条已执行轨迹的完整上下文”和“同一条轨迹但检索
正文不可见的上下文”，而不是删除检索 token 后重新排列序列。

### 2.2 两次教师前向

设学生采样到的完整序列为 $x_{1:L}$，第 $t$ 个待训练 token 的真实标签为
$y_t$，冻结教师为 $T$。教师对同一批 input ids、position ids 做两次相互
独立的前向计算。

#### Observed pass

完整上下文中的教师 logits 为：

$$
z^{\mathrm{obs}}_t = T(x_{1:L})_t.
$$

其中检索正文可被后续 token 访问。

#### Hidden pass

hidden pass 保持以下内容不变：

- token 的位置和序列槽位；
- position ids；
- padding 布局；
- 因果顺序；
- `<information>` 边界 token。

区别只在于构造一个逐样本的 4D additive causal attention mask：所有标记为
evidence 的 token 作为 key column 被屏蔽。因此任何 query 都无法读取检索正文，
但序列长度和位置编码不发生变化。hidden pass 的 logits 记为：

$$
z^{\mathrm{hid}}_t = T(x_{1:L};\ \text{evidence key columns blocked})_t.
$$

两次前向不共享 KV cache，避免把 observed 状态中的证据信息泄露到 hidden
状态。

### 2.3 证据残差教师分布

对每个 policy-token 行，ER-OPD 构造：

$$
z^{\mathrm{ER}}_t(\alpha)
  = z^{\mathrm{obs}}_t
    +\alpha\left(z^{\mathrm{obs}}_t-z^{\mathrm{hid}}_t\right)
  = (1+\alpha)z^{\mathrm{obs}}_t-\alpha z^{\mathrm{hid}}_t.
$$

其中

$$
\Delta z_t=z^{\mathrm{obs}}_t-z^{\mathrm{hid}}_t
$$

表示检索证据引起的教师 logit 变化。最终教师目标为：

$$
q_{\mathrm{ER},\alpha}(y\mid x_{\leq t})
  =\operatorname{softmax}(z^{\mathrm{ER}}_t(\alpha)).
$$

$\alpha\geq0$ 控制证据残差的强度：$\alpha=0$ 退化为普通 observed OPD，
$\alpha=1$ 精确恢复原先的 $2z^{\mathrm{obs}}-z^{\mathrm{hid}}$，更大的值进一步
强化证据使教师发生改变的方向。默认值为 1，因此旧实验行为不变。训练时只需要
目标分布对学生实际采样 token $y_t$ 的 log-prob：

$$
\log q_{\mathrm{ER}}(y_t\mid x_{\leq t}).
$$

### 2.4 ER-OPD 蒸馏优势

学生 rollout 阶段已经记录学生旧策略对相同 token 的 log-probability：

$$
\log p_{S}(y_t\mid x_{\leq t}).
$$

ER-OPD 使用 OPD 的 K1 sampled reverse-KL 信号：

$$
A^{\mathrm{ER}}_t
  =\log q_{\mathrm{ER}}(y_t\mid x_{\leq t})
   -\log p_{S}(y_t\mid x_{\leq t}).
$$

当前 ER-OPD 使用 `advantage_mode=token`，即每个 policy token 使用自己的
优势值，不做未来 token 累积。默认不做 advantage whitening，也不做
`clip_value` 截断。

### 2.5 与 GRPO outcome reward 的组合

ER-OPD 可以有两种运行模式。

#### 纯 ER-OPD

令 GRPO 系数为 0：

$$
A_t=A^{\mathrm{ER}}_t.
$$

此时每个问题只需要一条 on-policy trajectory，并且要求使用一个全局 PPO
mini-batch，以便对所有有效 policy token 做一致的全局均值更新。

#### ER-OPD + vanilla GRPO

当前 `train_er_opd.sh` 使用该模式：同一个问题采样多条轨迹（当前默认
`n_agent=8`），对最终答案 reward 做组内标准化，得到每条轨迹的 GRPO outcome
advantage $A^{\mathrm{GRPO}}_t$。最终优势为：

$$
A_t=\lambda_{\mathrm{distill}}A^{\mathrm{ER}}_t
   +c_{\mathrm{GRPO}}A^{\mathrm{GRPO}}_t.
$$

当前 launcher 的主要取值为：

```text
lambda_distill = 1.0
grpo_reward_coef = 1.0
n_agent = 8
use_gated_distillation = false
RCE = disabled
SOD = disabled
```

ER-OPD 本身不使用 sigmoid gate、RCE entropy gate、SOD step-wise weight 或
DGPO selective guidance；这些是仓库中的其他实验分支，不属于当前 ER-OPD 主方法。

### 2.6 策略更新与 token mask

训练更新流程如下：

```text
学生生成 Search-R1 多轮轨迹
        ↓
执行搜索动作并拼接检索 observation
        ↓
记录 evidence_mask、loss_mask 和学生 old_log_probs
        ↓
教师 observed pass：得到 z_obs
教师 hidden pass：屏蔽 evidence key columns，得到 z_hidden
        ↓
z_ER = 2 * z_obs - z_hidden
        ↓
计算 log q_ER(y_t) - log p_student(y_t)
        ↓
与组归一化 GRPO outcome advantage 相加（可选）
        ↓
使用 PPO actor objective 更新学生
```

mask 的作用可以概括为：

- **`evidence_mask`**：标记完整上下文中哪些 token 是检索正文，用于构造 hidden
  attention mask；
- **`loss_mask`**：保留 assistant 的 reasoning、search action、answer 等策略
  token，排除检索 observation；
- **`opd_distillation_mask`**：ER-OPD 的蒸馏范围，默认覆盖所有 assistant
  token，包括协议标签；
- **padding**：所有相关 mask 和统计均排除 padding。

### 2.7 数值与工程实现

为保证 ER 目标的稳定性和可复现性，当前实现包含以下约束：

- 学生和教师必须使用完全一致的 tokenizer vocabulary、added tokens 和 special
  tokens；
- ER target 使用完整共享 vocabulary 做 `log_softmax`，不能只在候选 token
  子集上归一化；
- logits、log-softmax、熵和 sampled-token extraction 在目标计算阶段使用
  FP32；
- token 行维度可以按 `target_token_chunk_size` 分块，分块只降低峰值显存，
  不改变 vocabulary 维度上的归一化；
- hidden attention mask 使用 SDPA 支持的逐样本 4D additive mask，因此
  `ref.attn_implementation=sdpa` 且 sequence parallel size 必须为 1；
- 默认要求 rollout temperature 为 1.0，避免教师目标与学生采样分布之间引入
  额外的温度缩放；
- 默认一个 rollout batch 只做一个 PPO epoch，且 actor 不额外使用 KL loss 或
  entropy regularization。

严格控制实验使用 FP32 教师参数和 FP32 teacher forward。当前优化版
`train_er_opd.sh` 明确使用 FP16 教师参数和 paired forward 以降低显存/计算开销，
但仍在 FP32 中完成证据残差构造和完整词表归一化；这种模式是显式的速度-精度
折中，不等价于严格 FP32 目标。

### 2.8 可观测指标

ER-OPD 训练中建议重点记录：

- `opd/observed_teacher_entropy`：完整证据上下文下教师熵；
- `opd/hidden_teacher_entropy`：屏蔽证据后的教师熵；
- `opd/er_logprob_shift`：ER 目标相对于 observed teacher 的 sampled-token
  log-prob 变化；
- `opd/selected_logit_residual`：采样 token 的
  $z^{\mathrm{obs}}-z^{\mathrm{hid}}$；
- `opd/evidence_residual_alpha`：本次运行使用的残差强度；
- `opd/evidence_context_token_fraction`：上下文中检索正文 token 比例；
- `opd/evidence_trajectory_rate`：包含检索证据的轨迹比例；
- `opd/distillation_advantage`：未乘系数的 ER-OPD advantage；
- `opd/grpo_advantage` 与 `opd/combined_advantage`：GRPO 分量及最终训练信号。

## 3. 代码对应关系

- 启动配置：[`train_er_opd.sh`](../train_er_opd.sh)
- OPD/ER-OPD 训练主循环：[`verl/trainer/ppo/ray_trainer.py`](../verl/trainer/ppo/ray_trainer.py)
- 教师 paired forward 与目标提取：[`verl/workers/actor/dp_actor.py`](../verl/workers/actor/dp_actor.py)
- 证据 hidden attention mask 与 ER 分布计算：[`verl/utils/evidence_residual.py`](../verl/utils/evidence_residual.py)
- observation tokenization 与 `evidence_mask` 生成：[`search_r1/llm_agent/generation.py`](../search_r1/llm_agent/generation.py)
- OPD advantage 与 GRPO 组合：[`verl/trainer/ppo/core_algos.py`](../verl/trainer/ppo/core_algos.py)

## 4. $\alpha$ 消融

启动脚本通过 `OPD_EVIDENCE_RESIDUAL_ALPHA` 接收 $\alpha$，同时应为每个运行设置
独立实验名，避免 W&B run 和 checkpoint 目录混淆：

```bash
for alpha in 0 0.5 1 1.5 2; do
  OPD_EVIDENCE_RESIDUAL_ALPHA="$alpha" \
  EXPERIMENT_NAME="eropd-grpo-1B-alpha-${alpha}" \
  ./train_er_opd.sh
done
```

其中 $\alpha=0$ 仍保留 observed/hidden 两次教师前向，因而与其他组计算协议一致；
它的目标分布则严格等价于 observed OPD。若使用 entropy-matched control，应针对
每个 $\alpha$ 独立校准并保存对应的校准产物。
