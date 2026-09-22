# 1 Introduction

搜索智能体需要根据检索结果修正判断：找到关键事实之后，后续推理与回答应当向事实支持的方向变化。[Search-R1](https://arxiv.org/abs/2503.09516v5) 等工作已通过结果奖励训练模型交替进行推理和检索。在策略蒸馏（On-Policy Distillation，OPD）进一步利用教师在学生自身轨迹上的预测提供稠密监督。然而，教师读取了正确证据，并不保证其最终偏好已经转向正确答案。本文关注一个具体问题：**当检索证据已经推动教师向正确方向更新，但教师仍然偏好错误答案时，如何利用这部分尚未完成纠错的更新来构造蒸馏目标？**

这一问题需要区分两个量：教师在完整上下文中的最终偏好，以及直接访问检索证据所带来的偏好变化。前者决定标准 OPD 所匹配的目标，后者描述该目标对检索证据可见性的响应。在学生生成的推理历史上，两者可能并不一致：开放检索证据后，正确答案相对于错误答案的支持增加，但增幅仍不足以逆转教师原有的错误排序。此时，教师并非完全没有响应证据；有用的变化已经出现，完整目标却仍保留错误偏好。直接匹配该目标可能限制对这一状态的纠正，但这一目标层面的风险是否转化为学生学习瓶颈，需要进一步实验验证。

我们通过固定学生前缀的配对教师诊断，将这种现象具体化。对相同的 token 序列和位置，分别开放与屏蔽真实检索正文的直接注意力访问，并比较正确、错误候选在首个分歧 token 处的对数赔率。图 1(a) 展示六个预先固定的、证据充分但学生答案错误的案例；其中三例通过历史概率复核，另三例仅作 CPU FP32 敏感性分析。Hemingway 案例中，证据隐藏时的赔率为 −3.18，证据可见时提高到 −0.63，表明证据改变了方向却尚未逆转排序。在事后追加的常用姓名完整候选比较中，ER 将分数差进一步提高至 +1.91，而熵匹配对照仍为负（图 1(b)）。这一初步结果说明该状态及其可利用信号确实可以出现；它并不证明其发生率、自由生成纠错率或学生训练收益。原定完整姓名候选未发生翻转，反向更新案例也同时保留在分析中。

现有研究已经注意到学生轨迹上的教师监督可能不可靠。[SOD](https://arxiv.org/html/2605.07725v3) 根据逐步师生分歧调节蒸馏权重，以缓解工具交互中的状态漂移。本文进一步考察目标分布本身：当证据诱导的变化有用、而完整教师仍偏错时，能否据此调整其候选偏好？在固定状态上，对蒸馏项施加正的标量权重不会重排教师分布中的候选；这为目标重构提供了明确的研究动机。不过，重加权仍可能通过后续优化与轨迹变化改善学生行为，完整 SOD 还联合使用结果奖励，因此这一差异不能被解释为 SOD 必然无法纠错。

基于这一问题，我们提出证据残差在策略蒸馏（Evidence-Residual On-Policy Distillation，ER-OPD）。设冻结教师在证据可见和证据隐藏视图下的 logits 为 $z_t^{\mathrm{obs}}$ 与 $z_t^{\mathrm{hid}}$，ER-OPD 构造目标

$$
q_t^{\mathrm{ER}}=\operatorname{softmax}\!\left[z_t^{\mathrm{obs}}+\alpha\left(z_t^{\mathrm{obs}}-z_t^{\mathrm{hid}}\right)\right].
$$

该目标在完整教师分布的基础上外推其对证据可见性的响应，尝试将“朝正确方向移动但尚未越过决策边界”的偏好转化为更明确的监督。两次教师计算保持问题、学生历史、协议标签与 token 位置一致，残差用于监督策略生成的 token，并可与结果奖励联合训练。证据隐藏视图仍可能通过学生历史间接获得相关事实，因此该差分是对特定可见性干预的响应，不是对纯证据因果贡献的识别。ER 也不自动知道残差方向是否正确；如果开放证据反而增强错误偏好，外推会进一步放大这种偏差。

据此，本文将评估分为三个递进问题：该状态在独立轨迹中是否可重复出现；残差外推能否在合法答案别名、熵匹配和自由生成检验下改善纠错；这种改善能否在相同训练预算下转化为学生的问答能力。目前完成的 CPU 诊断提供了局部候选层面的正面实例，同时在两个预定错误案例的教师续写中未观察到新增纠错。因此，当前证据支持将“有用但不足的证据更新”作为待验证的机制假设，尚不支持将其写成已证实的主要训练瓶颈或稳定性能收益。

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
