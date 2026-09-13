# ER-OPD Derivation Package

## Target

为正式方法章节核对 ER-OPD 的三条公式链：证据残差教师目标、sampled reverse-KL 蒸馏优势，以及与 GRPO 的联合策略优化目标。

## Status

**COHERENT AFTER REFRAMING**

残差目标本身是方法定义，而不是从最优性定理推出的唯一目标；概率比解释和 sampled reverse-KL 梯度关系是精确恒等式。方法章节必须区分这两类陈述。

## Invariant Object

固定学生实际生成的搜索轨迹后，冻结教师在“证据可见”和“证据键值被遮蔽”两个视图下对同一策略 token 给出的全词表预测分布。

## Assumptions

- 轨迹由旧学生策略 $$\pi_{\theta_{\mathrm{old}}}$$ 采样。
- 教师 $$p_\phi$$ 在训练期间冻结。
- 两次教师前向使用完全相同的 token ids 和 position ids，且不共享 KV cache。
- hidden view 只屏蔽检索正文对应的 key/value 列；问题、生成历史、协议标签和位置结构保持不变。
- 教师目标在完整词表上归一化。
- 检索正文 token 不参与策略损失；学生生成 token 和默认协议标签参与训练。
- 证据残差只描述固定轨迹下“直接证据可见性”对应的预测差异，不主张识别检索证据的完整因果效应。

## Notation

- $$x$$：输入问题。
- $$\tau$$：学生与搜索环境交互得到的轨迹。
- $$y_t$$：第 $$t$$ 个学生生成 token。
- $$m_t^\pi$$：策略 token mask。
- $$e_j$$：第 $$j$$ 个序列 token 是否属于检索正文。
- $$z_t^{\mathrm{obs}},z_t^{\mathrm{hid}}$$：教师在证据可见/隐藏视图下的 logits。
- $$q_t^{\mathrm{ER}}$$：ER-OPD 教师目标。
- $$\alpha\ge 0$$：证据残差强度，默认 $$\alpha=1$$。

## Derivation Strategy

先定义配对教师视图，再将 logits 差定义为证据残差；随后把该残差加入 observed logits 形成目标分布，并将其等价改写为概率比。最后从 reverse KL 的策略梯度恒等式得到 sampled-token advantage，并与组归一化结果优势相加。

## Derivation Map

1. evidence mask 与 causal mask 定义 hidden teacher view。
2. 两个视图的 logits 差定义 residual；这一步是方法定义。
3. softmax 代数给出 observed distribution 与 evidence likelihood ratio 的精确乘积形式。
4. reverse KL 的 score-function 梯度给出 sampled-token 蒸馏优势。
5. 蒸馏优势与 GRPO 优势线性组合后进入 clipped policy objective。

## Main Derivation

### 1. Evidence-residual target

定义：

$$
\Delta z_t^{\mathrm{ER}}
=z_t^{\mathrm{obs}}-z_t^{\mathrm{hid}}.
$$

ER-OPD 目标 logits 定义为：

$$
z_t^{\mathrm{ER}}
=z_t^{\mathrm{obs}}+\alpha\Delta z_t^{\mathrm{ER}}
=(1+\alpha)z_t^{\mathrm{obs}}-\alpha z_t^{\mathrm{hid}}.
$$

因此：

$$
q_t^{\mathrm{ER}}
=\operatorname{softmax}(z_t^{\mathrm{ER}}).
$$

当 $$\alpha=0$$ 时退化为标准 observed-teacher OPD；当前正式配置取 $$\alpha=1$$，得到 $$2z_t^{\mathrm{obs}}-z_t^{\mathrm{hid}}$$。

### 2. Probability-ratio identity

令 $$p_t^{\mathrm{obs}}=\operatorname{softmax}(z_t^{\mathrm{obs}})$$，$$p_t^{\mathrm{hid}}=\operatorname{softmax}(z_t^{\mathrm{hid}})$$。对任意词表 token $$v$$，有精确恒等式：

$$
q_t^{\mathrm{ER}}(v)
\propto
p_t^{\mathrm{obs}}(v)
\left(
\frac{p_t^{\mathrm{obs}}(v)}{p_t^{\mathrm{hid}}(v)}
\right)^\alpha.
$$

归一化常数与 $$v$$ 无关。该式允许把 ER target 解释为 observed teacher 乘以证据可见性对应的 likelihood ratio，但不保证该 ratio 总是代表高质量证据。

### 3. Sampled reverse-KL signal

对固定前缀状态 $$s_t$$，考虑：

$$
\mathcal{D}_t(\theta)
=D_{\mathrm{KL}}\!\left(
\pi_\theta(\cdot\mid s_t)
\,\|\,
q_t^{\mathrm{ER}}(\cdot\mid s_t)
\right).
$$

在 $$\theta=\theta_{\mathrm{old}}$$ 处，利用 score-function identity 以及 $$\mathbb{E}_{y\sim\pi}[\nabla\log\pi(y)]=0$$：

$$
-\nabla_\theta\mathcal{D}_t(\theta)
\Big|_{\theta=\theta_{\mathrm{old}}}
=
\mathbb{E}_{y_t\sim\pi_{\theta_{\mathrm{old}}}}
\left[
\nabla_\theta\log\pi_\theta(y_t\mid s_t)
\left(
\log q_t^{\mathrm{ER}}(y_t\mid s_t)
-\log\pi_{\theta_{\mathrm{old}}}(y_t\mid s_t)
\right)
\right].
$$

因此，单样本 token 级优势为：

$$
A_{t}^{\mathrm{ER}}
=m_t^\pi
\left[
\log q_t^{\mathrm{ER}}(y_t\mid s_t)
-\log\pi_{\theta_{\mathrm{old}}}(y_t\mid s_t)
\right].
$$

### 4. Joint advantage

对同一问题采样一组轨迹，并将组归一化结果奖励记为 $$A_i^{\mathrm{GRPO}}$$。联合优势为：

$$
A_{i,t}
=\lambda_{\mathrm{ER}}A_{i,t}^{\mathrm{ER}}
+c_{\mathrm{RL}}m_{i,t}^\pi A_i^{\mathrm{GRPO}}.
$$

当前正式配置取 $$\lambda_{\mathrm{ER}}=c_{\mathrm{RL}}=1$$，且不使用额外门控。

## Remarks and Interpretation

- ER target 的定义可视为对 evidence-sensitive token 的方向性强化；该解释不等于理论上证明这是唯一或最优的目标。
- 概率比恒等式是精确的，不依赖 logits 的线性可分假设。
- sampled reverse-KL 关系在旧策略处成立；实际训练使用固定优势进行多轮 clipped policy update。

## Boundaries and Non-Claims

- 不声称 $$z_t^{\mathrm{obs}}-z_t^{\mathrm{hid}}$$ 是检索证据的完整因果效应。
- 不声称较大的 evidence residual 必然对应正确或相关的证据。
- 不声称 ER target 一定优于完整教师分布；该结论必须由标准 OPD、entropy-matched OPD 和噪声证据实验验证。

## Open Risks

- 学生在证据可见条件下生成的后续推理可能复述检索内容，因此 hidden teacher 仍可通过生成历史间接获得部分信息。
- ER target 可能比 observed teacher 更尖锐，必须使用熵匹配对照排除普通 sharpening 解释。
- 错误检索也可能产生较大 residual，需要通过检索质量分层确定方法边界。

