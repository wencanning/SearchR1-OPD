# 1 Introduction

在一条搜索轨迹中，真正改变模型行为的关键信息往往并不是模型生成的，而是搜索引擎返回的。一段新证据可能推翻当前假设、触发下一次查询，也可能使智能体停止搜索并给出答案。Agentic Search/RAG 的能力正来自这种“获得信息—更新判断—继续行动”的闭环。可是，在现有训练流程中，模型生成的推理、查询和答案是学习目标，检索正文却属于环境返回的观察，通常会被排除在策略损失之外。证据仍能通过后续生成和最终奖励间接影响优化，却缺少一条与其信息内容对应的显式信用通道。由此形成了一个搜索智能体特有的不对称：**智能体被训练去寻找证据，却很少直接从“证据如何改变其行为”中学习。**

[Search-R1](https://arxiv.org/abs/2503.09516) 证明了仅凭最终答案奖励也能端到端学习多轮搜索，后续工作则从过程奖励、教师指导和优化稳定性等方向进一步提升搜索策略，例如 [InForage](https://arxiv.org/abs/2505.09316)、[DGPO](https://aclanthology.org/2026.acl-long.1751/) 和 [SAPO](https://arxiv.org/abs/2603.10069)。这些方法回答了“整条轨迹是否成功”“某次搜索是否有用”或“当前策略更新是否可靠”，但仍未充分回答一个更细粒度的问题：**当一段证据已经进入上下文，它具体应当如何改变智能体接下来的推理与行动？** 终局奖励把整条轨迹压缩为一个结果，过程奖励通常又把丰富的检索内容压缩为一个标量。即使一次搜索找到了决定答案的关键事实，训练信号也很难指出学生应当修正哪一步推理、是否继续搜索，以及哪些答案 token 应得到更强支持。

在策略蒸馏（On-Policy Distillation，OPD）为弥合这一缺口提供了自然起点。它让学生先生成自己的轨迹，再由更强的教师在学生实际访问的状态上提供 token 级评价，从而同时保留 on-policy 探索和稠密监督。近期研究已经将 OPD 用于上下文知识内化、工具推理和长程智能体训练，例如 [OPCD](https://arxiv.org/abs/2602.12275)、[SOD](https://arxiv.org/abs/2605.07725) 和 [SEED](https://arxiv.org/abs/2607.14777)。然而，将标准 OPD 直接用于搜索轨迹时，教师通常只在包含全部检索结果的完整上下文上给出一个目标分布。检索证据当然会影响这个分布，但单一目标只呈现教师读取证据后的评价结果，并不显式标记其中哪些变化来自刚刚获得的检索观察。问题并不在于完整教师分布不值得学习；恰恰相反，它是一个强基线。问题在于，只使用这一个分布，会丢掉搜索轨迹中天然存在的另一份监督：在其余条件保持不变时，直接访问检索证据究竟为教师的后续预测增加了什么。

这种监督也不同于一般的特权上下文蒸馏。OPCD 和 SEED 等方法利用额外提示或事后技能帮助学生在测试时不依赖这些外部上下文；而检索证据是智能体通过自身搜索动作在轨迹中获得的环境观察，并且在训练和推理时都会继续存在。因此，我们的目标不是让模型在没有证据时记住证据，而是为“如何使用已经获得的证据”分配信用。这引出了本文的核心问题：**能否让搜索智能体不仅从最终结果中学习，还直接从检索证据对其后续决策产生的影响中学习？**

为此，我们提出**证据残差在策略蒸馏**（Evidence-Residual On-Policy Distillation，ER-OPD）。对于同一条学生 on-policy 搜索轨迹，我们使用冻结教师构造两个严格配对的评价视图：一个允许教师读取检索正文，另一个在保持 token 序列、位置结构和学生生成历史完全一致的条件下，仅阻断教师对检索正文的直接访问。两个视图的预测差异形成**证据残差**，刻画检索观察的直接可见性如何重塑教师对后续 token 的评价。ER-OPD 沿这一残差方向重新构造教师目标，增强那些在证据可见时获得更多相对支持的候选 token，并削弱相反方向的候选 token。由此，检索结果不再只是被动进入上下文，而会为后续推理、查询决策和最终答案提供稠密的 token 级监督。

ER-OPD 与结果强化学习承担互补角色：结果奖励回答智能体最终是否解决了问题，证据残差则刻画在教师评价下，已获得的信息如何改变后续策略。训练时，我们将证据残差产生的 sampled reverse-KL 优势与组归一化的任务优势联合使用；检索正文仍作为环境观察被排除在策略损失之外，所有更新均发生在学生自己访问的状态和生成的 token 上。该方法不需要检索相关性标注、人工过程标签或额外的测试时模块，也不要求教师替代学生生成轨迹。

我们在 **[N 个开放域问答数据集]** 和 **[M 种模型规模]** 上评估 ER-OPD。与仅使用结果奖励的 Search-R1/GRPO 相比，ER-OPD 将平均 **[EM/F1]** 提升 **[X.X]** 个百分点；相较于直接蒸馏完整教师分布的标准 OPD，ER-OPD 提升 **[Y.Y]** 个百分点，并在 **[当前最强可复现基线]** 下保持优势。目标熵匹配、随机遮蔽和证据打乱实验进一步检验收益是否仅来自 logit sharpening、额外教师计算或任意上下文扰动；按检索质量分层的分析则刻画该方法在有效证据与噪声证据下的适用边界。**[此处在完成多随机种子结果审计后填入最强且可复现的实验结论。]**

本文的贡献如下：

1. 我们揭示了搜索智能体训练中的一种结构性监督缺口：检索证据能够改变后续行为，却通常不会作为独立信号参与 token 级信用分配。
2. 我们提出 ER-OPD，通过同一学生轨迹上的证据可见与证据隐藏教师视图提取证据残差，并据此构造面向后续生成决策的全词表蒸馏目标。
3. 我们通过与结果强化学习、标准 OPD 和教师指导方法的系统比较，以及目标熵匹配、遮蔽干预和检索质量分层实验，验证 ER-OPD 的有效性并分析其适用边界。

# 3 Preliminaries

## 3.1 Agentic Search

我们考虑多轮 Agentic Search 场景。给定问题 $$x$$，学生策略 $$\pi_\theta$$ 通过与搜索环境交互生成回答。在第 $$k$$ 轮中，策略生成响应 $$a^k$$，其中可以包含自然语言推理、搜索查询或最终答案；当策略发起搜索时，环境返回检索 observation $$o^k$$，并将其加入后续生成的上下文。完整轨迹写为：

$$
\tau=(x,a^1,o^1,\ldots,a^K,o^K,a^{K+1}).
$$

其中，$$K$$ 表示轨迹中的搜索轮数，$$a^k$$ 和 $$o^k$$ 分别表示第 $$k$$ 轮的策略响应和环境 observation，$$a^{K+1}$$ 表示最终响应。将轨迹展开为长度为 $$L$$ 的完整 token 序列 $$u_{1:L}$$。对于任意由策略生成的位置 $$t$$，记 $$y_t=u_t\in\mathcal V$$ 为对应 token，$$s_t=u_{<t}$$ 为其前缀状态，其中可以同时包含先前的策略输出和检索 observation；$$\mathcal V$$ 表示模型词表，$$\mathcal T(\tau)$$ 表示轨迹中所有策略生成 token 的位置集合。环境返回的 observation 不属于 $$\mathcal T(\tau)$$，因此不参与策略损失。

## 3.2 Group Relative Policy Optimization

Group Relative Policy Optimization（GRPO）通过比较同一问题下多条采样轨迹的奖励来更新策略。给定问题 $$x$$，旧策略 $$\pi_{\theta_{\mathrm{old}}}$$ 采样 $$G$$ 条轨迹 $$\{\tau_i\}_{i=1}^{G}$$，第 $$i$$ 条轨迹的结果奖励记为 $$R_i=r(\tau_i)$$。令 $$\mathbf R=(R_1,\ldots,R_G)$$ 表示该组奖励，则其均值和标准差为：

$$
\bar R=\operatorname{mean}(\mathbf R),
\qquad
\sigma_R=\operatorname{std}(\mathbf R).
$$

其中，$$r(\cdot)$$ 是任务奖励函数，$$\bar R$$ 和 $$\sigma_R$$ 分别表示组内奖励的均值和标准差。GRPO 使用归一化奖励作为第 $$i$$ 条轨迹的组相对优势：

$$
\hat A_i^{\mathrm{GRPO}}
=
\frac{R_i-\bar R}{\sigma_R+\epsilon_A},
$$

其中，$$\epsilon_A$$ 是保证数值稳定的小常数。由于这里采用 outcome-level reward，同一条轨迹中的所有策略 token 共享优势 $$\hat A_i^{\mathrm{GRPO}}$$。

为简化记号，令 $$\mathcal T_i=\mathcal T(\tau_i)$$ 表示轨迹 $$\tau_i$$ 中所有策略 token 的位置集合。对于任意 $$t\in\mathcal T_i$$，其 importance ratio 定义为：

$$
\rho_{i,t}(\theta)
=
\frac{
\pi_\theta(y_{i,t}\mid s_{i,t})
}{
\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid s_{i,t})
}.
$$

其中，$$y_{i,t}$$ 和 $$s_{i,t}$$ 分别表示第 $$i$$ 条轨迹中位置 $$t$$ 的策略 token 及其前缀状态，$$\pi_\theta$$ 是当前策略。GRPO 的 clipped policy loss 为：

$$
\mathcal L_{\mathrm{GRPO}}(\theta)
=
-\mathbb E\!\left[
\frac{1}{G}\sum_{i=1}^{G}
\frac{1}{|\mathcal T_i|}
\sum_{t\in\mathcal T_i}
\min\!\left(
\rho_{i,t}(\theta)\hat A_i^{\mathrm{GRPO}},
\operatorname{clip}\!\left(
\rho_{i,t}(\theta),1-\epsilon_{\mathrm{clip}},1+\epsilon_{\mathrm{clip}}
\right)\hat A_i^{\mathrm{GRPO}}
\right)
\right].
$$

其中，$$\operatorname{clip}(\cdot)$$ 表示截断算子，$$\epsilon_{\mathrm{clip}}$$ 控制策略更新的截断范围，期望对训练问题及其 on-policy 轨迹取值。该目标使用组内相对奖励提供轨迹级学习信号，而无需额外训练价值模型。

## 3.3 On-Policy Distillation

On-Policy Distillation（OPD）使用冻结教师在学生自己访问的状态上提供稠密监督。对于由学生策略采样的轨迹，标准 OPD 目标定义为：

$$
\mathcal L_{\mathrm{OPD}}
=
\mathbb E_{\tau\sim\pi_\theta}\!\left[
\frac{1}{|\mathcal T(\tau)|}
\sum_{t\in\mathcal T(\tau)}
\left(
\log\pi_\theta(y_t\mid s_t)
-\log\pi_{\mathrm{teacher}}(y_t\mid s_t)
\right)
\right].
$$

其中，$$\mathcal L_{\mathrm{OPD}}$$ 表示 OPD 的训练损失，$$\pi_{\mathrm{teacher}}$$ 表示参数固定的教师策略，$$\pi_\theta$$ 表示学生策略，$$y_t$$ 是学生实际生成的 token，$$s_t$$ 是对应的学生访问状态。该目标是学生与教师之间 reverse KL 的 sampled-token 估计，并只作用于 $$t\in\mathcal T(\tau)$$ 的策略 token。标准 OPD 直接使用教师在完整上下文上的分布；ER-OPD 将在此基础上进一步提取由检索正文直接可见性引起的教师预测变化。

# 4 Method

本节介绍证据残差在策略蒸馏（Evidence-Residual On-Policy Distillation，ER-OPD）。如图 **[Figure X]** 所示，ER-OPD 在同一条学生 on-policy 搜索轨迹上构造证据可见和证据隐藏两种教师评价，并利用二者的预测差异形成 evidence-aware distillation target。下面依次介绍配对教师评价、证据残差目标和训练目标。

## 4.1 Paired Teacher Evaluation

对于 Preliminaries 中定义的学生轨迹，冻结教师 $$\pi_{\mathrm{teacher}}$$ 执行两次评价。在 **evidence-observed view** 中，教师读取完整轨迹；在 **evidence-hidden view** 中，教师无法访问 `<information>` 与 `</information>` 之间的检索正文。两个视图使用相同的 token 序列和位置编码，并保留问题、学生生成历史及协议标签。对于任意策略 token $$t\in\mathcal T(\tau)$$，两个视图产生的全词表 logits 记为：

$$
z_t^{\mathrm{obs}}
=f_\phi(s_t;M^{\mathrm{obs}}),
\qquad
z_t^{\mathrm{hid}}
=f_\phi(s_t;M^{\mathrm{hid}}).
$$

其中，$$f_\phi$$ 表示冻结教师的前向函数，$$M^{\mathrm{obs}}$$ 和 $$M^{\mathrm{hid}}$$ 分别表示 evidence-observed view 与 evidence-hidden view 的可见性设置，$$z_t^{\mathrm{obs}},z_t^{\mathrm{hid}}\in\mathbb R^{|\mathcal V|}$$ 是教师在位置 $$t$$ 上输出的全词表 logit 向量。因此，两个教师预测的差异只来自检索正文是否对教师直接可见。具体的 attention-mask 构造和数值实现见附录。

## 4.2 Evidence-Residual Teacher Target

在固定轨迹上，我们将两个教师视图之间的 logit 差定义为证据残差：

$$
\Delta z_t^{\mathrm{ER}}
=z_t^{\mathrm{obs}}-z_t^{\mathrm{hid}}.
$$

该残差刻画检索正文的直接可见性如何改变教师对下一 token 的预测。ER-OPD 以 evidence-observed teacher 为基准，沿证据残差方向构造目标分布：

$$
\pi_{\mathrm{teacher},t}^{\mathrm{ER}}
=
\operatorname{softmax}\!\left(
z_t^{\mathrm{obs}}+\alpha\Delta z_t^{\mathrm{ER}}
\right)
=
\operatorname{softmax}\!\left(
(1+\alpha)z_t^{\mathrm{obs}}-\alpha z_t^{\mathrm{hid}}
\right),
$$

其中，$$\pi_{\mathrm{teacher},t}^{\mathrm{ER}}$$ 表示位置 $$t$$ 上的 evidence-residual teacher target，$$\operatorname{softmax}(\cdot)$$ 将 logit 向量转换为词表概率分布，$$\alpha\ge 0$$ 控制证据残差强度；当 $$\alpha=0$$ 时，该目标退化为标准 OPD。令 $$\pi_{\mathrm{teacher},t}^{\mathrm{obs}}=\operatorname{softmax}(z_t^{\mathrm{obs}})$$、$$\pi_{\mathrm{teacher},t}^{\mathrm{hid}}=\operatorname{softmax}(z_t^{\mathrm{hid}})$$，则对任意词表 token $$v$$，有：

$$
\pi_{\mathrm{teacher},t}^{\mathrm{ER}}(v)
\propto
\pi_{\mathrm{teacher},t}^{\mathrm{obs}}(v)
\left(
\frac{\pi_{\mathrm{teacher},t}^{\mathrm{obs}}(v)}{\pi_{\mathrm{teacher},t}^{\mathrm{hid}}(v)}
\right)^\alpha
.
$$

其中，$$\propto$$ 表示对右侧结果在词表 $$\mathcal V$$ 上归一化。该式将 ER target 解释为使用 evidence likelihood ratio 对完整教师分布进行重加权：读取检索正文后获得更高相对支持的 token 被增强，反之则被削弱。该残差只描述固定轨迹上与证据直接可见性相关的预测变化，并不假设检索内容一定正确。

## 4.3 Training Objective

对于学生策略生成的轨迹，ER-OPD 使用证据残差教师目标替代标准 OPD 中的教师分布。其蒸馏损失定义为：

$$
\mathcal L_{\mathrm{ER}}(\theta)
=
\mathbb E_{\tau\sim\pi_\theta}\!\left[
\frac{1}{|\mathcal T(\tau)|}
\sum_{t\in\mathcal T(\tau)}
\left(
\log\pi_\theta(y_t\mid u_{<t})
-\log\pi_{\mathrm{teacher},t}^{\mathrm{ER}}(y_t\mid u_{<t})
\right)
\right].
$$

其中，$$\mathcal L_{\mathrm{ER}}$$ 表示 ER-OPD 的 sampled reverse-KL loss，$$\pi_{\mathrm{teacher},t}^{\mathrm{ER}}$$ 是位置 $$t$$ 上的证据残差教师目标。该损失只作用于 $$t\in\mathcal T(\tau)$$ 的策略 token，因此环境返回的检索 observation 不参与优化。

与 SOD 类似，我们将稠密的蒸馏损失与基于结果奖励的 GRPO 损失结合，得到最终训练目标：

$$
\mathcal L_{\mathrm{ER\text{-}OPD}}(\theta)
=
c_{\mathrm{RL}}\mathcal L_{\mathrm{GRPO}}(\theta)
+\lambda_{\mathrm{ER}}\mathcal L_{\mathrm{ER}}(\theta).
$$

其中，$$c_{\mathrm{RL}}$$ 和 $$\lambda_{\mathrm{ER}}$$ 分别控制结果强化学习和证据残差蒸馏的强度。$$\mathcal L_{\mathrm{GRPO}}$$ 提供轨迹级结果反馈，$$\mathcal L_{\mathrm{ER}}$$ 则在学生访问的状态上提供由检索证据诱导的稠密 token 级监督。实际训练使用 Preliminaries 中的 clipped on-policy surrogate 估计该目标；具体实现和超参数见实验设置与附录。
