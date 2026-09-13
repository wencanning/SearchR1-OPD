# ER-OPD 论文叙事与引言写作建议

> 文献范围：截至 2026 年 8 月 29 日公开的论文与预印本；其中除特别注明外，多数 2026 年工作仍是 arXiv 预印本。本文档结合当前仓库中的 ER-OPD 实现撰写，实验数字均保留为占位符，不能在完成结果审计前写成事实。

## 1. 一句话结论

最适合 ER-OPD 的论文故事不是“把 OPD 用到 Search-R1”，也不是泛泛的“让教师更加可靠”，而是：

> **现有 OPD 蒸馏教师在完整轨迹上的总体偏好；ER-OPD 蒸馏的是检索证据究竟让教师的偏好发生了什么变化。**

英文可以凝练为：

> **ER-OPD distills not only what the teacher prefers, but what retrieved evidence changes in that preference.**

这条叙事把三个部分自然连起来：

1. Agentic Search/RAG 的关键变量是模型主动获得的检索证据；
2. 普通 OPD 的教师分布混合了语言先验、格式偏好、问题信息和证据信息；
3. ER-OPD 通过“证据可见/证据隐藏”的配对教师前向，显式提取证据引起的 logit 残差，并将其转化为稠密的 token 级训练信号。

因此，论文应围绕一个科学问题展开：

> **在搜索智能体的学生轨迹上，我们能否将教师的通用偏好与检索证据带来的增量偏好分离，并只强化后者？**

---

## 2. 2026 年 OPD 文献脉络

2026 年 OPD 研究的主线已经从“是否使用 on-policy 数据”转向三个更细的问题：教师看到什么、何时应该相信教师、如何在多轮智能体轨迹上稳定地提供监督。[OPD Survey](https://arxiv.org/abs/2604.00626) 也将不确定性感知反馈和 agent-level distillation 列为重要开放方向。

| 工作 | 核心机制 | 解决的问题 | 与 ER-OPD 的关系 |
|---|---|---|---|
| [OPCD](https://arxiv.org/abs/2602.12275) | 学生不看上下文，教师看特权上下文，在学生采样轨迹上做反向 KL | 将上下文知识内化到无上下文学生 | 证明“不同上下文视图上的 OPD”有效，但没有显式分解证据增量，也不针对内生检索证据 |
| [OPSD](https://arxiv.org/abs/2601.18734) | 同一模型的特权推理分支指导普通分支 | 自蒸馏高质量推理 | 属于 privileged-information OPD；ER-OPD 的区别应落在检索证据干预与残差目标构造上 |
| [Privileged Information Distillation](https://arxiv.org/abs/2602.04942) | 将特权信息分支蒸馏到普通策略 | 多轮 agent 的特权信息迁移 | 是需要引用的概念性上位工作，不宜声称 ER-OPD 首次使用 privileged views |
| [SOD](https://arxiv.org/abs/2605.07725) | 根据轨迹偏离程度逐步衰减教师信号 | 错误工具调用造成状态分布漂移后，教师监督失真 | SOD 回答“何时信教师”；ER-OPD 回答“教师偏好的哪一部分来自检索证据” |
| [Guided-OPD](https://arxiv.org/abs/2606.15912) | 训练早期混入教师生成的 turn，再逐步退火到纯学生轨迹 | 缓解多轮 agent 的 OOD drift | 改变轨迹生成过程；ER-OPD 保持学生 on-policy 轨迹不变，只改变教师目标 |
| [SDAR](https://arxiv.org/abs/2605.15155) | 特权技能分支、自蒸馏门控与 agentic RL 联合训练 | 将高层技能内化到策略 | 与“RL + 条件分支蒸馏”较近，但条件变量是技能提示，不是检索观察的局部反事实 |
| [OPID](https://arxiv.org/abs/2606.26790) | 从成功轨迹提炼 hindsight skills，再比较有/无技能上下文下的动作概率 | 将轨迹经验转化为层次化技能 | 同样使用上下文诱导的 log-prob shift；ER-OPD 必须强调检索证据、教师 logit 残差和新目标分布 |
| [SEED](https://arxiv.org/abs/2607.14777) | 当前策略生成 hindsight skills，在普通/技能增强上下文下重打分，并以置信门控结合 GRPO | 自演化技能蒸馏 | 是机制上的重要近邻；不能把“same action, different context rescoring”本身作为唯一创新 |
| [EDGE-OPD](https://arxiv.org/abs/2605.23493) | 用 token evidence mask 只更新被特权上下文支持的位置 | 减少无依据的 token 蒸馏 | 其“evidence”是监督 token 选择；ER-OPD 的 evidence 是检索观察，并用于构造分布残差 |
| [DEAR](https://arxiv.org/abs/2606.22830) | 从内部推理链中识别 decision-evidence tokens | 聚焦关键推理 token | 证据概念不同，但标题和术语接近，相关工作中应主动消歧 |
| [SA-OPD](https://arxiv.org/abs/2608.03632) | 比较完整输入与去掉整个 prompt 时的 teacher–student divergence，筛除低 grounding gap 且高分歧的 token | 过滤由模板、格式或输入无关先验引起的误导性教师信号 | **最直接的新颖性威胁**。它将对比值用于过滤 token；ER-OPD 只隐藏检索内容，并用 teacher-only residual 直接构造新目标分布 |

### 对文献格局的判断

不建议再使用以下宽泛贡献表述：

- “首个使用上下文差分的 OPD”；
- “首个利用特权信息指导 agent 的 OPD”；
- “首个解决教师不可靠问题的 OPD”；
- “首个 evidence-aware OPD”。

这些表述分别会受到 OPCD/OPSD、SDAR/OPID/SEED、SOD/SA-OPD 和 EDGE-OPD 的直接挑战。

可以 defend 的更窄、也更准确的表述是：

> **我们研究 agentic RAG 中由检索观察引起的教师偏好增量，并提出一种 retrieval-specific counterfactual target construction：在保持问题、动作前缀、协议 token 和位置完全相同的前提下，仅阻断检索证据的注意力可见性，用两个教师 logits 的残差构造 OPD 目标。**

最终是否能写“first”，仍建议在投稿前再做一次逐公式查新；当前更稳妥的措辞是 “we introduce” 或 “we study”，而不是 “we are the first”。

---

## 3. 2026 年 Agentic Search/RAG 文献脉络

Agentic Search/RAG 的近期工作主要在优化四个环节，而“如何从教师信号中提取证据的边际贡献”尚未成为主线。

### 3.1 基础范式：从静态 RAG 到多轮搜索策略

[Search-R1](https://arxiv.org/abs/2503.09516) 将搜索调用、阅读检索结果和最终回答统一为多轮强化学习轨迹，并对环境返回的检索内容进行 loss masking。它确立了 ER-OPD 所依赖的基本训练范式，但其主要监督仍来自最终答案奖励。

### 3.2 稳定优化与小模型训练

- [DGPO（ACL 2026）](https://aclanthology.org/2026.acl-long.1751/) 用冷启动蒸馏和 RL 阶段的选择性教师指导缓解紧凑模型的稀疏奖励与不稳定训练，是 ER-OPD 最重要的系统级基线之一。
- [Search-R1++](https://arxiv.org/abs/2602.19526) 系统研究 prompt、reward 和 policy optimizer，显示搜索 agent 的结论对训练 recipe 高度敏感。
- [SAPO](https://arxiv.org/abs/2603.10069) 将性能瓶颈定位到 GRPO 中的分布漂移和重要性采样，并通过条件 token-level KL 改善优化稳定性。

这组工作意味着：论文不能只与原始 Search-R1 比较。至少要证明 ER-OPD 的收益在强 optimizer/teacher-guidance baseline 下仍然存在。

### 3.3 检索质量、检索器与效率

- [InForage](https://arxiv.org/abs/2505.09316) 用中间检索质量奖励解决终局奖励难以分配到搜索决策的问题。
- [CoSearch](https://arxiv.org/abs/2604.17555) 联合训练推理 agent 与文档 ranker，指出固定 retriever 本身可能成为瓶颈。
- [Test-Time Strategies for Agentic RAG](https://arxiv.org/abs/2603.12396) 通过查询上下文化和去重改善搜索质量与调用效率。
- [Rethinking Agentic RAG](https://arxiv.org/abs/2605.27123) 探索由 LLM 生成结构化逻辑检索条件，以减少对复杂向量检索后端的依赖。

这些工作主要改变“检索什么”“怎样检索”或“如何奖励检索”。ER-OPD 的位置不同：它不直接训练 retriever，也不需要额外检索标签，而是从已经返回的观察中构造 token 级学习信号。

---

## 4. ER-OPD 应该怎样讲故事

### 4.1 冲突：普通 OPD 的教师分布不是纯粹的证据信号

在学生采样的搜索轨迹上，普通 OPD 使用教师在完整上下文中的分布：

$$
q_t^{\mathrm{OPD}} = \operatorname{softmax}\!\left(z_t^{\mathrm{obs}}\right).
$$

但这个分布同时编码了多种因素：

$$
z_t^{\mathrm{obs}}
\approx
z_t^{\mathrm{prior}}
+ z_t^{\mathrm{question}}
+ z_t^{\mathrm{trajectory}}
+ z_t^{\mathrm{evidence}}
+ \text{interactions}.
$$

这里的加法只是说明性分解，不应在论文中声称神经网络 logits 真的线性可分。核心问题是：即使教师高度偏好某个 token，也不能知道这个偏好究竟来自检索证据，还是来自教师原有知识、语言习惯或输出模板。

对 agentic RAG 来说，这不是普通的蒸馏噪声。模型的目标正是学习“何时搜索、如何读取、如何让后续推理依赖新获得的证据”。如果教师监督无法显式区分 evidence-dependent 与 evidence-independent preference，检索观察仍只是被动上下文，而不是具有明确 credit 的训练信号。

### 4.2 关键洞察：问教师“如果没有看到这段检索结果，你还会这样预测吗？”

对同一条学生轨迹进行两次独立教师前向：

- observed view：教师可以注意到完整检索证据；
- hidden-evidence view：只阻断 `<information>...</information>` 内证据 token 作为 attention keys/values，但保留问题、学生动作前缀、协议标签、token id 和 position id。

由此得到证据残差：

$$
\Delta z_t^{\mathrm{evidence}}
= z_t^{\mathrm{obs}} - z_t^{\mathrm{hid}}.
$$

它不是严格意义上的可识别因果效应，因此正文宜称为 **counterfactual evidence residual** 或 **evidence-conditioned logit contrast**，不要直接写成 “the causal effect of evidence”。

### 4.3 方法高潮：不是过滤教师，而是构造一个证据增强教师

令一般形式为：

$$
z_t^{\mathrm{ER}}
= z_t^{\mathrm{obs}}
+ \alpha\left(z_t^{\mathrm{obs}}-z_t^{\mathrm{hid}}\right),
$$

$$
q_t^{\mathrm{ER}}
= \operatorname{softmax}\!\left(z_t^{\mathrm{ER}}\right).
$$

默认实现取 $$\alpha=1$$，即：

$$
q_t^{\mathrm{ER}}
= \operatorname{softmax}\!\left(2z_t^{\mathrm{obs}}-z_t^{\mathrm{hid}}\right).
$$

这个公式可以给出一个比“logit 做减法”更清楚的概率解释：

$$
q_t^{\mathrm{ER}}(v)
\propto
p_t^{\mathrm{obs}}(v)
\left(
\frac{p_t^{\mathrm{obs}}(v)}{p_t^{\mathrm{hid}}(v)}
\right)^{\alpha}.
$$

也就是说，ER-OPD 从完整上下文教师分布出发，再乘上一个 evidence likelihood ratio。某个 token 在看到证据后相对更可能，就被强化；若其概率主要来自无证据时已经存在的先验，则不会获得同等强化。

在学生采样 token $$y_t$$ 上，训练信号为：

$$
A_t^{\mathrm{ER}}
= \log q_t^{\mathrm{ER}}(y_t)
- \log \pi_{\theta_{\mathrm{old}}}(y_t\mid s_t).
$$

再与任务结果奖励产生的 GRPO advantage 联合：

$$
A_t
= c_{\mathrm{RL}}A_t^{\mathrm{GRPO}}
+ \lambda_{\mathrm{ER}}A_t^{\mathrm{ER}}.
$$

检索观察 token 本身不参与 policy loss；ER 信号只监督学生生成的思考、搜索动作、答案和必要协议 token。

### 4.4 与最接近工作的本质区别

| 方法族 | 对比的上下文 | 对差异的用途 | 核心目标 |
|---|---|---|---|
| OPCD/OPSD | 有/无特权上下文 | 让学生匹配有上下文教师 | 内化知识或推理 |
| OPID/SEED/SDAR | 有/无 hindsight skill | 形成 skill-induced guidance，常配合门控 | 内化高层技能 |
| SA-OPD | 完整 prompt/无 prompt | 识别并过滤疑似输入无关的 token | 避免 spurious teacher signal |
| **ER-OPD** | **检索证据可见/仅检索证据隐藏** | **将 teacher-only logit residual 变成新的全词表目标分布** | **强化检索证据带来的增量偏好** |

这一表格应在相关工作末尾或方法开头以文字形式讲清楚。

---

## 5. 推荐的整篇论文叙事

### 5.1 开场问题

“Agentic RAG 已经能够通过 RL 学会搜索，但终局奖励很难告诉模型哪一段检索结果真正改变了后续决策。OPD 提供稠密教师监督，却仍把证据作用与教师先验混在同一个分布中。”

### 5.2 诊断实验先于方法

论文最好先给出一个小型但直观的 motivating study，而不是直接上公式：

1. 对同一批 Search-R1 学生轨迹，分别计算 $$p^{\mathrm{obs}}$$ 和 $$p^{\mathrm{hid}}$$；
2. 比较检索命中与检索未命中轨迹的 residual norm、token KL 或 sampled-token log-ratio；
3. 展示普通 OPD 高置信 token 中，有多少在隐藏证据后几乎不变；
4. 用一个案例标出哪些后续实体、关系或答案 token 只在证据可见时被明显提升。

如果这个诊断现象足够强，引言中的问题就从直觉变成可测量事实。

### 5.3 方法

方法保持简单：学生 on-policy rollout、冻结教师配对重打分、证据残差目标、sampled reverse-KL advantage、与 GRPO 联合训练。不要让实现细节淹没核心概念；4D attention mask、独立 KV cache 和 FP32 target construction 放入实现细节或附录。

### 5.4 实验回答三个问题

- **RQ1：效果。** ER-OPD 是否优于 Search-R1/GRPO、普通 OPD、DGPO 和 SOD，并在不同数据集与模型规模上成立？
- **RQ2：机制。** 收益是否真的来自 evidence residual，而不是简单降低目标熵、增加教师计算或更强的 logit sharpening？
- **RQ3：边界。** 当检索结果无关、错误或含有干扰信息时，ER-OPD 会怎样？

### 5.5 结论边界

如果 ER-OPD 只在检索命中时明显有效，论文仍然可以成立，但应把结论写成“有效地利用有支撑的检索观察”，并讨论未来的 residual gating。若在 noisy retrieval 上也稳定，才可以进一步主张 robustness。

---

## 6. 引言应如何组织

建议写成六段，不要在第一段堆所有相关工作。

### 第 1 段：任务价值与基本矛盾

说明 agentic RAG 让模型通过多轮搜索获取外部知识，但 compact agents 仍受稀疏终局奖励、长程 credit assignment 和检索噪声影响。落点是：“关键不是让模型生成更多搜索，而是让它学会其决策如何依赖搜索得到的证据。”

### 第 2 段：现有 OPD 为什么看起来合适

介绍 OPD 在学生自己访问的状态上提供 token 级教师信号，能够避免纯离线蒸馏的分布错位。简要引用 OPCD、SOD、DGPO 等，承认多轮稳定性和教师选择已经被积极研究。

### 第 3 段：指出尚未解决的 retrieval-specific blind spot

这是整篇引言最重要的一段。教师在完整轨迹上的高概率不等于 evidence-grounded preference。完整教师分布可能主要来自参数知识、问题线索、响应模板或前文轨迹。现有 OPD 因而告诉学生“教师喜欢什么”，却没有显式回答“检索结果改变了什么”。

### 第 4 段：核心洞察和方法

引出 evidence-hidden counterfactual。强调只遮蔽检索内容，其他 token、前缀和位置保持不变。给出一个核心公式和 likelihood-ratio 解释，避免在引言里展开全部训练目标。

### 第 5 段：经验结果

只能写已经被实验支持的三类结果：总体准确率、机制性对照、噪声/泛化分析。现在先保留占位符，不能填估计值。

### 第 6 段：贡献

贡献应分别对应问题、方法和证据，不要写成三种措辞重复描述 ER-OPD。

---

## 7. 英文引言初稿

下面的版本可以作为论文正文起点。方括号中的数字和数据集必须在实验完成后替换；相关工作引用可在进入 LaTeX 时转换为 BibTeX citation keys。

### Introduction

Large language models are increasingly trained to seek information rather than answer solely from their parametric knowledge. In agentic retrieval-augmented generation (RAG), a model iteratively reasons, issues search queries, reads retrieved passages, and decides when it has enough evidence to answer. Reinforcement learning has made this behavior trainable end to end, but compact search agents remain difficult to optimize: the final answer reward is sparse, the consequence of a search action may appear many turns later, and retrieved passages vary substantially in relevance and reliability. The central learning problem is therefore not merely to encourage more search, but to teach the policy how its subsequent decisions should depend on the evidence it has acquired.

On-policy distillation (OPD) offers a promising source of dense supervision. Instead of imitating teacher-generated trajectories, OPD evaluates actions sampled from the student on the states that the student actually visits. Recent work has used privileged context, step-wise weighting, teacher-guided rollouts, and skill-conditioned branches to improve reasoning and multi-turn agents. Search-agent training has likewise benefited from selective teacher guidance and more stable policy optimization. These advances address important failures caused by distribution shift and unreliable supervision. However, they generally treat the teacher distribution under the available context as a single target.

This treatment leaves a retrieval-specific blind spot. A teacher may assign high probability to an action because it is supported by the retrieved passage, but it may also do so because of parametric knowledge, cues in the question, response-format priors, or the preceding trajectory. Standard OPD does not distinguish these sources: it distills what the teacher prefers after retrieval, without identifying what retrieval changed in that preference. This distinction matters for agentic RAG, where retrieved observations are endogenous, variable-quality information acquired by the policy itself. Conflating evidence-dependent and evidence-independent preferences can provide dense supervision while still giving the retrieval event no explicit credit.

We introduce **Evidence-Residual On-Policy Distillation (ER-OPD)**, a retrieval-specific counterfactual target for search agents. For each student-generated trajectory, a frozen teacher performs two independent evaluations over the same token sequence and positions. In the observed view, the teacher attends to the complete retrieved passages. In the hidden-evidence view, only the content inside retrieval observations is blocked as attention keys and values, while the question, action prefix, protocol tokens, and positional structure are unchanged. If $$z_t^{\mathrm{obs}}$$ and $$z_t^{\mathrm{hid}}$$ denote the resulting logits, ER-OPD constructs

$$
q_t^{\mathrm{ER}}
= \operatorname{softmax}\!\left(
z_t^{\mathrm{obs}}
+ \alpha(z_t^{\mathrm{obs}}-z_t^{\mathrm{hid}})
\right).
$$

Equivalently, the target reweights the observed-context teacher by an evidence likelihood ratio, $$q_t^{\mathrm{ER}}(v) \propto p_t^{\mathrm{obs}}(v)(p_t^{\mathrm{obs}}(v)/p_t^{\mathrm{hid}}(v))^{\alpha}$$. Tokens whose support increases when evidence is available are amplified, whereas preferences already present without the retrieved content receive less relative emphasis. We convert this target into a sampled reverse-KL advantage on policy-generated tokens and combine it with the outcome-based RL advantage. Retrieval observations remain excluded from the policy loss.

Across [N] question-answering benchmarks and [M] compact model scales, ER-OPD improves exact-match accuracy by [X.X–Y.Y] points over outcome-only RL and by [A.A] points over standard OPD. The gains persist against [DGPO/SOD/strongest verified baseline] and are largest on trajectories where retrieved passages contain answer-supporting evidence. Controlled comparisons against temperature sharpening and entropy-matched OPD show that the improvement cannot be explained by a sharper teacher distribution alone. Under [distractor/shuffled/irrelevant] retrieval, [state the verified robustness result without exaggeration]. Together, these results show that explicitly modeling the teacher's evidence-induced preference shift provides useful credit for training search agents.

Our contributions are threefold:

1. We identify and measure a retrieval-specific limitation of standard OPD: its teacher target conflates preferences induced by retrieved evidence with preferences already present in the teacher.
2. We propose ER-OPD, which uses paired evidence-observed and evidence-hidden teacher evaluations to construct an evidence-residual target on student-generated search trajectories.
3. We provide controlled experiments that separate evidence attribution from generic logit sharpening and evaluate when the residual signal helps or fails under varying retrieval quality.

---

## 8. 必须补齐的实验，否则故事站不稳

### 8.1 主结果基线

最低要求：

- Search-R1/GRPO；
- vanilla OPD，即 $$q=\operatorname{softmax}(z^{\mathrm{obs}})$$；
- DGPO 或其可复现的 selective teacher guidance 版本；
- SOD；
- 若实现成本允许，加入 SA-OPD 风格 filtering，直接回应最近工作。

### 8.2 排除“只是 sharpening”的关键对照

当 $$\alpha>0$$ 时，$$z^{\mathrm{obs}}+\alpha(z^{\mathrm{obs}}-z^{\mathrm{hid}})$$ 通常会改变目标熵。审稿人很可能认为收益只是 target sharpening。必须加入：

- 温度缩放 OPD；
- 与 ER-OPD 平均熵匹配的 OPD；
- $$\alpha\in\{0,0.5,1.0,1.5,2.0\}$$ 的消融，其中 $$\alpha=0$$ 就是 vanilla OPD；
- 随机遮蔽同长度 token、遮蔽问题、打乱检索证据等干预对照。

### 8.3 证据质量分层

至少按以下类别报告：

- answer-supporting retrieval；
- relevant but insufficient retrieval；
- irrelevant retrieval；
- distractor or contradicted retrieval；
- zero-search trajectories。

关注指标除 EM/F1 外，还应包括：

- residual norm 或 $$D_{\mathrm{KL}}(p^{\mathrm{obs}}\|p^{\mathrm{hid}})$$；
- sampled-token evidence log-ratio；
- 搜索轮数与成功率；
- 引用/答案是否可由检索文本支持；
- 不同 token 类型上的 ER advantage，例如 query、reasoning、final answer。

### 8.4 计算成本

ER-OPD 比普通 OPD 多一次教师前向。应报告 wall-clock、峰值显存和吞吐量，并说明两次前向不共享 KV cache 是为了保证 attention intervention 正确。若性能收益不大，计算代价会成为主要审稿问题。

### 8.5 统计可靠性

- 至少 3 个随机种子；
- 报告均值、标准差或 bootstrap 置信区间；
- 对主比较做 paired significance test；
- 训练曲线同时展示均值和跨种子区间，不能只选最好 run。

---

## 9. 论文标题与首页图建议

推荐标题按“记忆点—稳妥性”排序：

1. **What Did Retrieval Change? Evidence-Residual On-Policy Distillation for Search Agents**
2. **Distilling Evidence-Induced Preferences for Agentic RAG**
3. **Evidence-Residual On-Policy Distillation for Compact Search Agents**

首页图建议只表达一个对比：

- 左：vanilla OPD 将完整教师分布全部传给学生，其中混合 prior 与 evidence effect；
- 中：同一学生轨迹上的 observed/hidden-evidence 两个教师视图；
- 右：两者残差形成 evidence likelihood ratio，并与 RL advantage 合并。

图中最好配一个具体 token 示例，例如检索到实体关系后，正确实体 token 的概率从 hidden view 的低值上升到 observed view 的高值；不要只画抽象模块框。

---

## 10. 最终写作建议

论文最有说服力的版本应避免把 ER-OPD 包装成又一个复杂的 OPD recipe，而应把它写成一个清楚的测量与学习原则：

> **A retrieval observation deserves credit to the extent that it changes the teacher's preference on the student's own trajectory.**

方法简单反而是优势。真正决定论文强度的不是再增加门控、RCE 或更多 loss，而是用诊断实验和严格对照证明：

1. 普通 OPD 的确包含大量 evidence-independent teacher preference；
2. ER residual 与证据支持程度相关；
3. ER-OPD 的收益不能由目标熵、额外计算或普通 teacher sharpening 解释；
4. 方法在错误检索下的行为边界被诚实地量化。

如果这四点成立，故事会从“一个工程变体”上升为“agentic RAG 中如何给检索证据分配 token-level credit”的方法论文。
