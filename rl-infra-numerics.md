# 02 · 训练推理数值一致性与低精度 RL

> **状态：正文已补全 · CPU 教学实验已运行。** 2026-10-29 核对技术报告与固定提交源码；2026-11-02 整理发布，补齐符号定义、数值机制与路由重放的解释。本文区分数学恒等式、工程近似、作者实测与本地教学实验；尚未复现 GPU kernel 性能或大模型 RL 收敛。

[返回笔记目录](../) · [下载实验脚本](../scripts/rl-infra-numerics-lab.py) · [实验原始结果](../assets/rl-infra-numerics/results.json)

这一篇要解决一个实际排障问题：**同一个 checkpoint，rollout 记录的 logprob 与 trainer 重算的 logprob 为什么不同？这个差异何时会改变梯度，如何定位，又该修到什么程度？** 读完应能画出三种概率的来源、追踪一次 loss 的分母、设计一组隔离变量的对照实验。

先记住三个判断。权重版本只是执行状态的一部分；FP4 是数值格式，不能完整描述执行过程；一致性指标改善与训练质量、端到端速度改善，需要分别验证。后文将逐项解释。

## 第 1 章 三种概率与两类偏差

### 1.1 先把 rollout、old 与 current 分开

本篇的概率推导采用一个明确的任务设定：**单轮问答，输入一个 prompt，模型生成一段 response；每生成一个 token，就是执行一次 action。** 这里不包含历史对话，也没有中途插入的用户消息或工具返回。后文的 token 概率、PPO ratio 和序列概率比，都沿用这个设定。

**从 prompt 到回答。** prompt 是给模型的问题及必要的固定指令，经 tokenizer 编码后记作 $x$。token 是模型处理文本的基本单位，在程序中用整数 ID 表示，不一定对应一个汉字或一个完整单词。模型根据当前可见的前缀预测下一个 token，将抽中的 token 追加到回答，再继续预测，直到满足停止条件。

| 符号 | 定义 |
| --- | --- |
| $\mathcal V$ | token 词表，即候选 token ID 的集合 |
| $x=(x_1,\ldots,x_m)$ | **固定的 prompt 部分**，共 $m$ 个 token；不包含生成的回答 |
| $t$ | 回答内部的 token 位置，从 1 开始；不是训练更新次数 |
| $a\in\mathcal V$ | 任意一个候选 token，也就是候选 action |
| $a_t$ | 第 $t$ 步实际抽中的 token，即该步执行的 action |
| $a_{<t}=(a_1,\ldots,a_{t-1})$ | 回答中已经生成的前 $t-1$ 个 token，不含当前的 $a_t$ |
| $h_t=(x,a_{<t})$ | 预测 $a_t$ 时可见的完整前缀：固定 prompt 与已生成回答的拼接 |
| $y=(a_1,\ldots,a_T)$ | 最终生成的整段 response，长度为 $T$；它由 $T$ 个 token actions 组成 |

下标 $<t$ 表示“位置在 $t$ 之前”，不是不等式 $a<t$。$h_t$ 中的括号表示序列拼接。因此，**$x$ 始终不变，$h_t$ 随回答增长**：第一步尚未生成回答，$h_1=x$；生成 $a_t$ 后，把它接到 $h_t$ 后面就得到 $h_{t+1}$。

**用一次回答对照这些符号。** 假设 prompt 是“1+1 等于几？”，回答依次生成了下面五个 token：`[答, 案, 是, 2, <eos>]`。这是人为设定的 token 切分，用于讲解；真实切分取决于 tokenizer，`<eos>` 表示结束 token。

| 步骤 | 已生成回答 $a_{<t}$ | 完整前缀 $h_t$ | 选中的 $a_t$ |
| --- | --- | --- | --- |
| $t=1$ | 空序列 | $x$ | `答` |
| $t=2$ | `[答]` | $x$ 后接 `答` | `案` |
| $t=3$ | `[答, 案]` | $x$ 后接 `答案` | `是` |
| $t=4$ | `[答, 案, 是]` | $x$ 后接 `答案是` | `2` |
| $t=5$ | `[答, 案, 是, 2]` | $x$ 后接 `答案是2` | `<eos>` |

在第 4 步，模型面对的候选 $a$ 可以是 `2`、`3` 或其他 token；这条样本实际选中的 $a_4$ 是 token `2`。接下来要比较的，就是不同执行路径给**同一个前缀 $h_4$ 下、同一个 token $a_4$**分配的概率。

**同一个 token，为什么会有三种概率？** rollout 是采样引擎生成回答的过程，sampler 根据模型输出与采样规则抽取 token；trainer 则对记录下来的回答重算概率，并据此更新模型。在这个流程中，需要区分：

| 记号 | 含义与来源 | 训练时的处理 |
| --- | --- | --- |
| $q(a\mid h_t)$ | rollout 实际抽取 token 的行为分布，包含采样规则与推理执行路径的影响 | 记录下来的行为概率视作常量 |
| $p_0(a\mid h_t)$ | 本次优化开始前，trainer 用选定的冻结参数重算的分布，即 old | 冻结，作为更新的参照 |
| $p_\theta(a\mid h_t)$ | trainer 用当前参数 $\theta$ 计算的分布，即 current | 参与求梯度与参数更新 |

竖线 $\mid$ 读作“给定……”。$q(\cdot\mid h_t)$ 表示整个候选分布，$q(a\mid h_t)$ 是候选 $a$ 的概率，$q(a_t\mid h_t)$ 则是已选中 token 的概率。下标 0 是本次优化的参考标记，不是 token 位置；$\theta$ 是当前可训练参数，冻结参考参数记作 $\theta_0$。

例如，对前缀“prompt + 答案是”与 token `2`，可以有 $q(a_4\mid h_4)=0.60$、$p_0(a_4\mid h_4)=0.58$、$p_\theta(a_4\mid h_4)=0.62$。这是教学假设值，说明三侧评价的是同一事件。对齐实验必须固定这份前缀和 token；若两侧各自生成了不同文本，就不能把后续概率直接当作同一事件比较。

![图 1 三种概率的来源 | 本文原创示意。两条 ratio 的分母不同；图中 loss 为 PPO 加 TIS 的常见分支，不覆盖所有算法。](../assets/rl-infra-numerics/policies.svg)

**差异从哪里来？** 一是生成数据与重算概率时的权重版本可能不同；二是同版本下的训练、推理执行路径可能不同；三是 sampler 还可能施加 temperature、top-p 等变换。只记录 checkpoint 名称不足以区分这三类原因。

为把它们分开，记 rollout 使用的权重版本为 $v$，trainer 的冻结参考版本为 $u$。上标 $R$ 表示 rollout 引擎，$L$ 表示 learner/trainer；$u,v$ 是版本标签，$R,L$ 是引擎标签。$p_v^R$ 是采样变换前的 rollout 模型分布，$p_v^L$ 是同版本的 trainer 分布，$p_u^L$ 对应上面的 $p_0$，$q_v$ 对应实际行为分布 $q$。

下面各概率都在同一个 $(a_t\mid h_t)$ 上取值，为简洁省略该条件。$\log$ 表示自然对数，$\exp$ 是它的逆运算。

**式 1：把滞后、执行差异和采样变换分开**

$$
\log p_u^L-\log q_v=\underbrace{\log p_u^L-\log p_v^L}_{\text{weight lag}}+\underbrace{\log p_v^L-\log p_v^R}_{\text{execution gap}}+\underbrace{\log p_v^R-\log q_v}_{\text{sampling transform}}.
$$

第一项是权重版本变化，第二项是同版本执行差异，第三项是采样变换。它是加减中间项得到的代数恒等式，不能把左侧的全部差异都称为“量化误差”。若训练端也施加采样变换，比较时还要明确两侧采用的是变换前还是变换后的分布；下一节专门讨论这个问题。

本节的 prompt、response、prefix 记号及三种策略来自 [slime 官方说明的固定版本](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/docs/en/advanced/policy-mismatch.md#L9)。论文记号对照与实现细节见第 6 章。

### 1.2 返回的 logprob 是否对应实际采样分布？

固定生成步骤 $t$，将前文的 $h_t$ 简写为 $h$。模型在此前缀下输出 logits 向量 $z$：词表 $\mathcal V$ 中每个 token 对应一个未归一化分数。$a$ 是待求概率的候选 token，$z_a$ 是它的分数；$b$ 是遍历候选的求和索引，$z_b$ 是相应分数。这里 $a,b$ 都是 token 索引，不是生成步数。

设温度 $\tau>0$，允许采样的非空集合为 $S(h)\subseteq\mathcal V$；该集合还取决于采样配置，无筛选时就是整个词表。固定集合时，温度越小，概率越集中于高分 token。$\mathbf{1}_{a\in S(h)}$ 在 $a$ 属于集合时取 1，否则取 0；$\exp(u)=e^u$ 将分数转为正权重。一个简化的行为分布是：

**式 2：采样分布的归一化范围**

$$
q(a\mid h)=\frac{\mathbf{1}_{a\in S(h)}\exp(z_a/\tau)}{\sum_{b\in S(h)}\exp(z_b/\tau)}.
$$

分母对所有允许 token 的权重求和，使概率总和为 1。**$q(a\mid h)$ 是概率，实际采样分布对应的 logprob 是 $\log q(a\mid h)$。** 本节要核对的是：引擎返回的 logprob 是否对应这个经过采样处理的分布。

例如原始模型概率是 $(0.6,0.3,0.1)$，截断后只允许前两项，则实际采样概率是 $(2/3,1/3,0)$。若日志仍返回原始 log-softmax，就不能把它当作 $\log q$，用于后文式 4 的行为校正 ratio。若 temperature=0、使用 greedy，行为分布在选中动作上退化为点质量，也不能直接当成原始 softmax。

核对到的 [vLLM sampler 固定源码](https://github.com/vllm-project/vllm/blob/61a6f8138fd81f89467af762dc1b74674f544300/vllm/v1/sample/sampler.py#L141) 中，`raw_logprobs` 在 logits processors 与采样前计算；后续可由 processed 路径替换返回结果。因此字段名字不足以确定概率语义，还要追到 sampler 的具体分支。SGLang 也有原始 logprob、temperature 与确定性计算路径的分支，不能假定所有版本、后端都相同。[SGLang sampler](https://github.com/sgl-project/sglang/blob/f6649627836699f76a3054d0812c05ac332932c0/python/sglang/srt/layers/sampler.py#L231)

工程上先用 $\tau=1$、无 top-k/top-p 截断、无 penalty/grammar 的设置隔离算术差异，再逐项恢复真实采样配置。完整日志至少保存 token IDs、位置、response mask、权重版本和 sampling config；有支持集重放时还应保存实际保留集合。重放集合定义的是“在这个集合上的分布”，并不会自动恢复被截断掉的目标概率质量。

**排障时先用 $\tau=1$ 去掉温度缩放这个变量，方便定位算术差异。** 它不是说正式训练必须用 1；而且即使温度为 1，top-p、量化和 kernel 差异仍然可能改变分布。

例如，Dressage 的 [SWE-Gym 训练脚本](https://github.com/Accio-Lab/Dressage/blob/3e3142fe8ea07e4504c3b20a936a4c201a3de44c/examples/scripts/run_dressage_swegym_qwen3.5_4b_claude_code_sync_4_node.sh#L204) 默认使用 `ROLLOUT_TEMPERATURE=1.0`，可通过环境变量修改；[代理源码](https://github.com/Accio-Lab/Dressage/blob/3e3142fe8ea07e4504c3b20a936a4c201a3de44c/dressage/proxy/server.py#L505) 则优先采用请求中的 `temperature`，未提供时才回退到配置值。因此，示例默认值不等于所有请求的实际采样温度。（核对版本：`3e3142f`。）

### 1.3 为什么 baseline 在错分布下会产生额外项

令 $s(a)=\nabla_\theta\log p_\theta(a\mid h)$。$\nabla_\theta$ 表示对模型参数求梯度，所以 $s(a)$ 是与参数同维度的梯度向量，称为 score function，不是奖励分数。$\mathbb E_p[f(a)]$ 表示按分布 $p$ 加权的平均，即 $\sum_{a\in\mathcal V}p(a\mid h)f(a)$。对固定前缀和正规化的可微正概率，有：

**式 3：score 的零均值来自采样分布**

$$
\mathbb{E}_{p_\theta}[s(a)]=\sum_a\nabla_\theta p_\theta(a)=0,\qquad \mathbb{E}_{q}[s(a)]\ne0\ \text{in general}.
$$

第一项为零，是因为 $p_\theta\nabla_\theta\log p_\theta=\nabla_\theta p_\theta$，而所有候选的概率之和恒为 1，对参数求导就是零。baseline $b(h)$ 是只依赖前缀、不依赖当前候选动作的标量参照值，计算这里的策略梯度项时将它视作常量；它与式 2 的求和索引 $b$ 无关。因此，在目标分布下减去它不改变期望梯度；换成 $q$ 采样却不校正时，期望中会多出 $-b\,\mathbb{E}_{q}[s]$。这说明数值差异可以改变更新方向，而不仅是使 loss 的显示值有一点不同。本章推导限定固定前缀；group advantage 的随机归一化、自包含 baseline 和长轨迹状态分布还需要另外分析。

### 1.4 PPO ratio 与行为校正 ratio 不能混为一谈

下面每个概率都在记录下来的 $(a_t,h_t)$ 上取值，并假设分母为正。$r_t(\theta)$ 是当前 trainer 与冻结参考的概率比；$\rho_t$ 是冻结参考与实际采样来源的概率比。它们都是当前 token 对应的标量，不是整段回答的概率比。

**式 4：两条 ratio**

$$
r_t(\theta)=\exp(\log p_\theta-\log p_0),\qquad \rho_t=\exp(\log p_0-\log q),\qquad r_t\rho_t=\frac{p_\theta}{q}.
$$

$r$ 衡量 trainer 相对于参考策略的更新；$\rho$ 衡量参考策略与采样来源的差别。在 $p_\theta=p_0$ 时，$r=1$，但 $\rho$ 仍可能很大。PPO 自身的 clipping 因而不会自动发现同版本执行错位。

对给定前缀，若目标有质量的动作在 $q$ 下都有正概率，则换测度成立。下式的 $f=f(a)$ 是任意待求期望的标量或向量函数，例如“优势乘 score”；$\rho=p_0(a\mid h)/q(a\mid h)$ 随动作变化。$w(\rho)$ 是人为设计的校正权重函数，不是模型参数。IS 是 importance sampling（重要性采样），TIS 是对重要性权重做截断的变体。

**式 5：精确 IS 与截断的偏差**

$$
\mathbb{E}_{p_0}[f]=\mathbb{E}_{q}[\rho f],\qquad \mathbb{E}_{q}[w(\rho)f]-\mathbb{E}_{p_0}[f]=\mathbb{E}_{q}[(w(\rho)-\rho)f].
$$

TIS 用受限权重换取较温和的尾部；它不是无偏修复。以 $w=\min(\rho,2)$ 为例，$\rho=4$ 的贡献被压到 2；区间拒绝规则 $w=\rho\mathbf{1}_{0.5\le\rho\le2}$ 则直接将它置零。后者也会丢弃过小的 ratio。阈值是实验配置，不能因为某篇教程使用 2 就当成通用最佳值。

一种常见的最小化 surrogate（替代优化目标）写成下式。$\mathcal L$ 是要最小化的标量损失；$A=A_t$ 是该 token 的优势估计，衡量它相对参照的好坏，这个策略损失中将其视为常量。$\epsilon_-,\epsilon_+\ge0$ 分别控制下、上裁剪幅度；$\operatorname{clip}(r,l,u)$ 把 $r$ 限制在区间 $[l,u]$，$\min$ 取两项中较小者。$r,\rho$ 省略了 token 下标。

**式 6：PPO 外乘冻结的行为校正**

$$
\mathcal{L}=-\mathbb{E}_{q}\!\left[\operatorname{sg}(w(\rho))\min\!\left(rA,\operatorname{clip}(r,1-\epsilon_-,1+\epsilon_+)A\right)\right].
$$

`sg` 表示 stop-gradient：**只停止对它包住的校正权重 $w(\rho)$ 求导，保留该权重的数值；不是停止整个损失的反向传播。** 这里 $\rho=p_0/q$ 来自冻结的参考概率与行为概率，$A$ 也在这条策略损失中作为常量。当前策略仍通过 $r(\theta)=p_\theta/p_0$ 参与求导，梯度沿分子 $p_\theta$ 传回模型参数。

以未触发裁剪的分支为例，记单个 token 的损失为 $\ell=-\operatorname{sg}(w(\rho))\,r(\theta)A$，则 $\nabla_\theta\ell=-w(\rho)A\,\nabla_\theta r(\theta)$。**只对 $r(\theta)$ 求导，校正权重和 $A$ 都作为系数乘在梯度上。** 例如 $w(\rho)=2$、$A=3$，梯度就是 $-6\nabla_\theta r(\theta)$，并没有丢掉校正权重；也不是把梯度传给 $A$。选中裁剪后的常数分支时，该 token 的这项梯度可以为零，这是 clipping 的作用。

窄屏公式中的 $\widehat r$ 是裁剪后的 $r$，$m$ 是取最小值后的项。该式说明分工，不代表任意异步数据下的精确 policy gradient；优势估计、clipping、状态分布与 reduction 都引入额外条件。实现必须追踪张量的来源，确认哪些量已冻结，不能仅看公式中的符号。[slime 算法说明](https://thudm.github.io/slime/advanced/policy-mismatch.html)

### 1.5 token-wise 校正遗漏了什么

**问题：校正当前 token，为什么还不够？** 前面的 $\rho_t=p_0(a_t\mid h_t)/q(a_t\mid h_t)$ 比较的是：在同一个前缀 $h_t$ 下，两种策略选择当前 token 的概率。它没有比较两种策略生成这个前缀的概率。即使不截断 $\rho_t$，这个区别也存在。

**两步例子。** 固定同一个 prompt $x$，假设回答只生成两个 token，第一步可选“甲”或“乙”。下面是教学假设值，不是论文实验数据：

| 事件 | 参考策略 $p_0$ | 实际采样策略 $q$ |
| --- | ---: | ---: |
| 第一步生成“甲” | 0.8 | 0.2 |
| 已生成“甲”后，第二步生成“好” | 0.5 | 0.5 |

第二步“好”的校正比是 $\rho_2=0.5/0.5=1$。但整段“甲好”的概率分别是 $0.8\times0.5=0.4$ 和 $0.2\times0.5=0.1$，相差四倍。差别来自第一步：采样策略生成“甲”的机会更少，第二步的 $\rho_2$ 看不到这一点。

**完整回答的校正需要连乘。** 对回答 $y=(a_1,\ldots,a_T)$，$T$ 为回答长度，自回归概率按各步条件概率相乘，因此有：

**式 7：完整回答的概率比**

$$
\frac{p_0(y\mid x)}{q(y\mid x)}=\prod_{t=1}^{T}\rho_t.
$$

$\prod$ 表示连乘；例中 $\rho_1=0.8/0.2=4$，所以 $\rho_1\rho_2=4$。**第一步的损失乘 $\rho_1$、第二步的损失乘 $\rho_2$，不等于第二步的损失乘 $\rho_1\rho_2$。** 这就是 token-wise 校正没有修正前缀分布的含义。[slime 固定版本的 TIS 说明](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/docs/en/advanced/policy-mismatch.md#L72) 明确区分了这两种校正。

**为什么不直接全部改成连乘？** 长回答的概率比连乘后，可能出现少数样本权重特别大的情况，使估计波动很大。$p_0(y\mid x)/q(y\mid x)=\prod_{t=1}^{T}\rho_t$ 是概率恒等式，不是把 PPO 改成序列乘积就能保证训练更好的建议。另外，重要性采样要求目标有概率的回答也能被 $q$ 采到；若 top-p 截断使某些目标回答的采样概率为零，连乘也无法补回这些从未出现的回答。具体来说，top-p 按概率从高到低保留累计概率达到设定阈值的候选集合，将集合外 token 的实际采样概率置为零，再对集合内概率重新归一化。一条回答只要在某一步需要选中被排除的 token，整条回答的 $q(y\mid x)$ 就为零，即使未作同样截断的参考策略仍给它正概率。

**有效样本量（Effective Sample Size，ESS）：检查权重是否过于集中。** 先给每条回答计算一个权重，再用这些权重计算 ESS。对同一个 prompt $x$ 的 $N$ 条回答，记第 $i$ 条为 $y_i$，长度为 $T_i$；$\rho_{i,t}$ 表示这条回答第 $t$ 个 token 的参考概率与行为概率之比。

这里采用未经截断的序列概率比作为回答权重：$w_i=p_0(y_i\mid x)/q(y_i\mid x)=\prod_{t=1}^{T_i}\rho_{i,t}$。也就是把同一条回答各个 token 的 $\rho$ 连乘；上面的“甲好”对应 $w_i=4\times1=4$。**$w_i$ 是第 $i$ 条回答的权重，前面的 $w(\rho)$ 则是对单个 token 的校正比进行截断等处理的函数，两者不是同一个记号。** ESS 本身不规定权重的来源，这里明确选用序列概率比。

有了这 $N$ 个非负权重，且至少一个非零，就可以计算 $\mathrm{ESS}=(\sum_i w_i)^2/\sum_i w_i^2$，其中 $\sum_i$ 对回答求和。100 条回答权重相等时，ESS 为 100；只有一条权重非零时，ESS 为 1。

对已经采到的回答，$q(y_i\mid x)>0$，原始权重 $w_i=0$ 需要 $p_0(y_i\mid x)=0$：例如参考策略 $p_0$ 也经过候选屏蔽或 top-p 截断，排除了回答中的某个 token，而采样策略 $q$ 在相同前缀下仍允许它；这与前面 $q=0$ 导致回答采不到的情况不同。若 $p_0$ 使用有限 logits、无屏蔽或截断的完整 softmax，则数学上权重为正，“只有一条权重非零”只是极端例子；一条权重远大于其他权重时，ESS 也会接近 1。

ESS 只衡量权重集中程度，不能证明校正无偏或训练有效。公式及局限见 [Art Owen《Monte Carlo》§9.3，式 (9.13)](https://artowen.su.domains/mc/Ch-var-is.pdf#page=11)。

## 第 2 章 浮点执行与 batch invariance

### 2.1 可复现性、批次不变性、跨引擎一致性

沿用前面的例子：问题是“1+1 等于几？”，已生成“答案是”，现在计算下一个 token `2` 的概率。若训练端与推理端给出的数值不同，怎样判断差异来自哪里？先固定同一份权重、前缀的 token 编号和位置，比较这一位置的 logits（全词表的未归一化分数），或 token `2` 的 logprob（log probability，对数概率）。**不要让两边各自生成不同回答后，再直接比较后续概率。**

**先说明“相同”指什么：**

- **生成文本相同**：两边都回答“答案是2”，仍可能给 `2` 不同的概率。
- **数值接近**：对应的 logits 或 logprob 误差在事先约定的容差内，需要报告误差与阈值。
- **逐比特相同**：在相同数值格式下，所比较数值的二进制表示完全相同，比“足够接近”更严格。

数学等价也不保证逐比特相同。例如，批量矩阵乘法后取第一项 `(A @ B)[0]`，与单独计算 `A[0] @ B[0]`，浮点结果可能不同。这是 [PyTorch 官方数值说明](https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html#batched-computations-or-slice-computations) 明确指出的限制。

**再说明固定了什么、改变了什么。** 下表沿用同一份权重与前缀，每次只改变一项；它是本文为排障整理的四个比较维度，不是文献中统一、互斥的四类定义。比较 logprob 时，还须统一温度、候选截断等概率处理规则。

| 比较维度 | 具体比较什么 |
| --- | --- |
| 同配置重复运行：检查可复现性 | 输入、硬件、软件版本和执行配置都不变，多次计算同一位置的 logits，结果是否相同？ |
| 改变批次：检查批次不变性（batch invariance） | 请求本身不变，单独运行、与其他请求一起运行，或改变批内顺序，同一位置的 logits 是否相同？ |
| 改变序列处理方式 | 一次处理整个前缀（prefill，预填充），与利用缓存逐 token 处理（decode，解码），到达同一前缀末尾时的 logits 是否相同？这里逐步输入固定 token，不重新采样。 |
| 改变引擎：检查跨引擎数值对齐 | 训练端与推理端在同一前缀下，给同一个 token `2` 算出的 logprob 是否相同？ |

这些维度有重叠。Thinking Machines 区分了[同配置下的确定性与批次不变性](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/#batch-invariance-and-determinism)；其 [batch-invariant attention](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/#batch-invariant-attention) 还要求结果不受序列分块、缓存及 prefill/decode 处理方式影响。因此，不能笼统说“批次不变性不包含 prefill/decode 一致性”。

**工具开关能保证到哪一步？** 固定随机种子控制随机数序列；`torch.use_deterministic_algorithms(True)` 为受支持操作选择确定性算法，已知非确定且无替代实现时会报错。它不保证跨版本、跨平台或跨引擎结果相同。[PyTorch 可复现性文档](https://docs.pytorch.org/docs/2.15/notes/randomness.html)

vLLM 的 `VLLM_BATCH_INVARIANT=1` 用于启用批次不变性。[官方功能文档](https://docs.vllm.ai/en/latest/features/batch_invariance/) 在 2026-11-01 核对时仍标为 beta；使用时须按实际版本核对模型、硬件和注意力实现的支持范围。跨训练／推理引擎还需对齐量化、舍入、累加顺序及缓存读写等细节。vLLM [2026-10-14 技术报告 §2、§4](https://vllm.ai/blog/2026-10-14-train-inference-consistency) 的 DeepSeek-V4-Flash 方案展示了这些约束，其对齐结果限于报告验证的配置，不能推广为任意框架组合的保证。

### 2.2 批次变化为什么可能改变浮点结果

同一个请求换了批次，输入内容没有变，计算结果为什么仍可能变化？先看一个只改变加法括号的例子，再把它连接到批量计算。

**同样三个数，先加哪两个会影响结果。** FP32（32-bit floating point，32 位单精度浮点数）能保留的有效数字有限，无法精确表示所有数。下面每一步加法都舍入到 FP32：

- $(10^8-10^8)+1$：先算 $10^8-10^8=0$，再算 $0+1$，得到 **1**。
- $10^8+(-10^8+1)$：括号内的精确结果是 $-99\,999\,999$，但舍入到 FP32 后变回 $-100\,000\,000$；再与 $10^8$ 相加，得到 **0**。

第二条路径中，加上的 1 在中间舍入时消失了，后面的相加无法把它恢复。这就是浮点加法不满足结合律的例子。[本地教学脚本](../scripts/rl-infra-numerics-lab.py) 的 `f32` 函数与 `run` 函数开头实现了上述计算；这是中央处理器（Central Processing Unit，CPU）上的舍入实验，不代表测量了某个图形处理器（Graphics Processing Unit，GPU）计算程序的行为。

**并行求和，会把“先加哪两个”变成分组问题。** 求和归约（sum reduction）就是把多个数合并成一个和。并行计算可以先分别算几组小和，再合并它们。例如，四个数可以按 $(a+b)+(c+d)$ 求和，也可以按 $(a+c)+(b+d)$ 求和；分组不同，中间舍入发生的位置就不同。**并行不必然产生随机结果：两套求和顺序可以各自稳定，却得到不同结果。** 这也是 Thinking Machines 区分[重复运行确定性与批次不变性](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/#batch-invariance-and-determinism) 的关键。

**批次大小，可能影响计算库选择哪种分组。** 用矩阵乘法 $C=AB$ 看这个联系：$A$ 的形状为 $M\times K$，$B$ 为 $K\times N$，输出 $C$ 为 $M\times N$。在一个模型线性层的例子中，$A$ 每行是一枚 token 的输入，$B$ 是固定权重；一起处理的 token 越多，$M$ 就越大。

每个输出元素是 $C_{ij}=\sum_{k=1}^{K}A_{ik}B_{kj}$。其中，$i,j$ 选择输出的行与列，$k$ 遍历需要累加的 $K$ 个乘积。分配并行工作时，可以：

- **沿 $M/N$ 分配**：让不同计算单元负责不同输出元素。
- **沿 $K$ 分块**：把同一个输出的求和拆给多个计算单元，各自算部分和，最后合并。

批次较小时，可并行计算的输出较少，计算库可能选择沿 $K$ 拆分来利用更多计算单元；批次较大时，则可能采用另一种方案。目标 token 的输入和权重虽然没变，它的乘积却可能换了累加顺序。关键不是“M/N 一定安全、K 一定不安全”，而是**每个输出的实际计算与舍入过程是否改变**。[Thinking Machines：矩阵乘法中的批次不变性](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/#batch-invariant-matrix-multiplication)

因此，这条因果链是：**批次大小变化 → 可能选择不同的计算程序（kernel）或分块方案 → 可能改变同一输出的累加顺序 → 可能改变浮点结果。** 不是每次换批次都会换算法，也不是每次换算法都会出现数值差异。[PyTorch 官方文档](https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html#batched-computations-or-slice-computations) 因而不保证数学等价的批量计算与逐项计算逐比特相同。固定累加方式仍可并行，但会约束部分优化选择，实际性能代价需要测量。

### 2.3 数值差异怎样影响概率比与训练

上一节解释了计算结果为什么可能不同。现在固定同一个前缀，追问：**logits 的差异怎样传到同一个 token 的概率，再影响训练使用的校正权重？**

**先看完整 softmax 的概率变化。** 第一套 logits 记为 $z$，第二套为 $z+\Delta z$；$i,j$ 都是词表中的 token 索引。第一套概率是 $p_i=\operatorname{softmax}(z)_i$，第二套是 $p'_i=\operatorname{softmax}(z+\Delta z)_i$。定义 $\Delta\log p_i=\log p'_i-\log p_i$，始终是“第二套减第一套”；本节用大写 $\Delta$ 表示两次计算的差值，用 $\partial$ 表示偏导，二者含义不同。

把 softmax 概率取自然对数，利用“商的对数等于对数之差”，可以展开为：

$$
\log p_i=\log\frac{e^{z_i}}{\sum_j e^{z_j}}=\underbrace{\log e^{z_i}}_{z_i}-\log\left(\sum_j e^{z_j}\right)=z_i-\log\left(\sum_j e^{z_j}\right).
$$

其中 $\sum_j$ 表示对所有候选 token 求和。要算 $\log p_i$ 的变化，只需分别算右边两项的变化：第一项是 $\Delta z_i$；第二项记为 $F(z)=\log\left(\sum_j e^{z_j}\right)$，按复合函数求导，有 $\frac{\partial F}{\partial z_j}(z)=\frac{e^{z_j}}{\sum_k e^{z_k}}=p_j$，其中 $k$ 同样遍历词表。**$p_j$ 始终表示第一套分布中 token $j$ 的概率；求导得到的分式恰好就是 softmax 概率，所以偏导数等于 $p_j$，并没有把 $p_j$ 重新定义为求导符号。**

对较小扰动，一阶泰勒展开就是**将每个变量的变化量乘以对应偏导数，再求和**，因此 $F(z+\Delta z)-F(z)\approx\sum_j\frac{\partial F}{\partial z_j}(z)\,\Delta z_j=\sum_j p_j\Delta z_j$。偏导数在原始 $z$ 处计算，最后一步用其计算结果 $p_j$ 代入。用第一项的变化减去第二项的变化，得到下式：

**式 8：logits 扰动如何改变 logprob**

$$
\Delta\log p_i=\log p'_i-\log p_i=\Delta z_i-\bigl[F(z+\Delta z)-F(z)\bigr]\approx\Delta z_i-\sum_j p_j\Delta z_j.
$$

一个 token 的 logprob 变化，既取决于自己的 logit 变化，也取决于其他候选变化后带来的归一化调整。$p_j$ 是**第一套分布**的概率权重；$\approx$ 表示忽略二阶及更高阶项。

- **全部同加一个常数**：例如 $(0,0)$ 变为 $(3,3)$，概率仍是 $(0.5,0.5)$。softmax 的分子、分母同时乘上 $e^3$ 并抵消，所以数学上精确不变，不只是一阶近似不变。
- **候选之间的相对分数改变**：例如 $(0,0)$ 变为 $(0.1,0)$，第一个 token 的概率变为 $e^{0.1}/(e^{0.1}+1)\approx0.5250$。因此，不能把任何 logits 差异都等同于概率差异。

**再接回训练中的校正比。** 沿用第 1 章：$h$ 是固定前缀，$a$ 是待比较的同一个 token，$p_0$ 是训练端更新前的参考分布，$q$ 是实际采样的行为分布。在两者对该 token 的概率都为正时，明确约定 $\Delta=\log p_0(a\mid h)-\log q(a\mid h)$，才有 $\rho=p_0(a\mid h)/q(a\mid h)=e^\Delta$。只有把第一套对应到 $q$、第二套对应到 $p_0$ 时，前面的 $\Delta\log p_i$ 才与这里的 $\Delta$ 同方向。[miniTE v1 §2、式 (2)](https://arxiv.org/html/2610.23928v1#S2) 也先约定训练端减采样端，再取指数。

例如 $\Delta=0.2$ 时，$\rho\approx1.2214$，即参考概率约为行为概率的 1.22 倍。差值很小时，$e^\Delta\approx1+\Delta$，并非任意微小误差都会被剧烈放大。$\rho$ 的变化可能改变第 1 章的校正权重 $w(\rho)$，从而改变该 token 对梯度的贡献；它与近端策略优化（Proximal Policy Optimization，PPO）的更新比 $r=p_\theta/p_0$ 是两回事，截断或裁剪规则不能混用。[slime 固定版本说明](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/docs/en/advanced/policy-mismatch.md#L55)

式 8 针对**完整 softmax**。若 $q$ 还经过温度缩放、top-p 候选截断等处理，计算校正比必须使用处理后的实际分布，不能直接把原始 logits 的差异套进来；此时 $\Delta$ 也可能包含采样规则差异，而不只是执行误差（见 1.2）。下一节在固定这些比较条件后，定位数值差异从哪里开始。

### 2.4 怎样逐步定位数值差异的来源

**先固定评分对象。** 取一份已有回答，让两侧读取相同的 token 前缀，计算其中下一个指定 token 的概率，不让它们重新采样回答。这称为固定序列评分（teacher-forced scoring），是前向评分过程，不执行参数更新。例如，前缀固定为“prompt + 答案是”，两侧都计算 token `2` 的概率。miniTE 的[冻结权重比较（§2、§3.1）](https://arxiv.org/html/2610.23928v1#S3.SS1) 使用了这种方法。

比较前，先回答三个问题：

1. **实际加载的权重版本是否相同？** 检查加载结果与同步状态，不能只看模型名称；评分期间不更新权重。
2. **输入 token、位置和注意力可见范围是否一致？** 核对分词器与对话模板产生的 token 序列；补齐长度用的填充 token，不应意外改变有效位置或可见上下文。
3. **比较的是同一个目标 token、同一种 logprob 吗？** 生成 $a_t$ 的概率应以 $x,a_{<t}$ 为条件，不能把 $a_t$ 自己也作为已知前缀。还要确认两侧返回的是原始模型概率，还是经过采样规则处理后的概率。

**再用四组对照缩小范围。** 先约定判据：要求逐比特相等，还是允许预先设定的绝对／相对误差。以下对照和后面的日志清单，是本文根据来源整理的排障建议，并非某个框架已有的统一接口。

| 比较方式 | 优先检查的方向 |
| --- | --- |
| 同配置重复评分 | 执行随机性、缓存或其他状态变化 |
| 固定目标请求，改变批次组成 | 批次相关的算子选择、分块或量化尺度 |
| 固定序列，比较整段处理与逐步处理 | 缓存、位置处理、不同阶段的计算路径 |
| 两个引擎按对应层比较 | 差异最早在哪个可比边界出现 |

这些是排查方向，不能仅凭某种现象直接断定原因。例如，整段与逐步处理的差异，可能同时涉及缓存和算子路径。

**找到边界后，再做相同输入重放。** 先对齐两侧张量所代表的 token、位置与布局，找到最早观察到差异的层或模块；再固定该模块的入口输入、权重及缓存等相关状态，比较两种实现。**相同入口输入可以排除模块之前传入的输入差异，但模块内部后续算子的输入仍可能不同。**

例如，一个模块包含“第一个全连接层 → 激活函数 → 第二个全连接层”。即使模块入口相同，前两步也可能产生不同中间值，因此第二个全连接层的输出差异不能全归因于它自身的矩阵乘法。miniTE [附录 C 的相同输入重放](https://arxiv.org/html/2610.23928v1#A3) 固定的是混合专家（Mixture of Experts，MoE）层入口，并明确说明了这一限制。若要单独检查第二个矩阵乘法，还要固定它实际接收的操作数与相关状态。

**最后缩减为可重复运行的最小例子。** 保存必要的权重与软件版本、硬件、输入 token 与位置、批次配置、精度与采样配置，以及比较位置和两侧数值。若线索涉及量化、专家选择或缓存，再补充尺度、所选专家或缓存内容。逐步去掉无关层和请求，每次确认差异仍可复现；调整实现后，先重跑这个例子，再回到原评分任务验证。[vLLM 技术报告 §5](https://vllm.ai/blog/2026-10-14-train-inference-consistency#5-lowering-engineering-cost-with-agent--skill)

最早观察到数值差异，只说明从这个边界开始两侧不再满足约定的相等标准，不自动意味着这里存在程序错误；还需判断它是实现缺陷，还是不同浮点计算方式带来的结果差异。

## 第 3 章 低精度 RL 与 QAT

### 3.1 “低精度”用在哪些环节

**“使用 FP4 训练”不是一个完整配置。** FP4（4-bit floating point）指 4 位浮点表示，但训练包含多个环节：前向计算根据输入算输出与损失，反向计算梯度，优化器根据梯度更新参数；rollout 则用模型生成训练样本。这些环节可以采用不同精度，连同一次矩阵乘法的输入、累加和输出，也要分别说明。[Transformer Engine 混合精度说明](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html#mixed-precision-training-a-quick-introduction)

说明一种方案时，可以按下面各项填写精度配置。这是本文整理的检查表：

| 环节或对象 | 需要说清什么 |
| --- | --- |
| 张量存储 | 哪些权重、激活（中间计算结果）以低精度保存？读入计算前是否转换格式？ |
| 前向运算输入 | 哪些层的哪些运算实际接收低精度操作数？ |
| 累加与输出 | 乘积用什么精度相加？结果以什么精度输出？ |
| 反向计算 | 计算梯度的运算用什么精度？梯度又以什么精度保存和累加？ |
| 参数更新 | 更新运算及用于积累更新的权重采用什么精度？若另存一份“主权重”，需说明其精度；并非所有实现都有独立副本。 |
| 优化器状态 | 动量、梯度平方的滑动平均等历史统计量，以什么精度保存？ |
| rollout | 生成样本时采用什么存储与计算配置？它与训练端是否对齐，要另外验证。 |

**伪量化用在训练或微调中，让模型提前适应量化误差。** 假设准备把模型以 4 位格式部署，直接量化后准确率下降，就可以在训练前向中加入伪量化（fake quantization）：把权重或激活量化，再立即反量化成浮点张量，交给后面的运算。反量化恢复的是量级和浮点表示，**不会恢复舍入时丢失的信息**，因此损失能反映量化误差，训练再据此调整权重。这个过程是量化感知训练（Quantization-Aware Training，QAT）的一种常见实现；反向怎样通过舍入操作，见 3.4。[NVIDIA 的 QAT 用途说明](https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/)

**把一次训练拆开看。** 以 BF16（bfloat16，16 位浮点格式）前向运算、模拟 FP4 量化的配置为例：

- **前向传播：** BF16 权重／激活 → 模拟 FP4 量化 → 反量化为 BF16 近似值 → BF16 矩阵乘法 → 输出 → 计算损失。
- **反向传播：** 从损失出发 → 逐层计算梯度 → 经过伪量化操作时采用 STE（Straight-Through Estimator，直通估计器）等替代梯度规则 → 得到参数梯度。
- **参数更新：** 优化器根据梯度更新权重，再进入下一次前向传播。

这里的量化和反量化都属于**前向**；反向运算的精度另行配置，参数更新也是单独一步。反量化后的 BF16 近似值用于前向矩阵乘法，不是等到反向才使用；实现也可以直接模拟量化—反量化的数值效果，无须实际存出 FP4 数据。STE 的具体作用见 3.4。

例如，下节的数值 $5.2$ 量化再恢复后变成 $6.0$；伪量化可以仍用浮点张量保存这个 $6.0$，让后续计算看到这份近似值。它的主要用途是改善最终量化模型的准确率，训练本身可能更慢。典型流程是：**浮点模型 → 加入伪量化并训练 → 转换为低位模型 → 交给部署后端执行**。[PyTorch 原文](https://pytorch.org/blog/quantization-aware-training/) 给出了训练与转换代码，以及对应的[流程原图（Figure 2）](https://pytorch.org/wp-content/uploads/2024/11/fg2-2.png)。

![原图截图：量化误差怎样进入训练循环 | 来源：NVIDIA《How Quantization Aware Training Enables Low-Precision Accuracy Recovery》（2025-09-11），Figure 2。2026-11-01 从 Safari 截图，仅裁去浏览器界面与外围留白，图内内容未改动；右侧绿色箭头连接前向、损失、反向和权重更新。](../assets/rl-infra-numerics/nvidia-qat-loop-screenshot.jpg)

**读图：** 先看右边的 QAT Training Loop（量化感知训练循环）：前向使用量化后的近似值 → 计算训练损失 → 反向计算梯度 → 更新高精度权重 → 再做下一次前向。伪量化就放在第一步，使误差参与后续的参数调整。图中写的是 Fake/Real，因为 QAT 也可以采用真实低精度前向；它并不限定只能伪量化。图中的 STE 对应上面反向传播中的替代梯度规则，详见 3.4。[NVIDIA 原文与图注](https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/)

**原生低精度计算用在实际运算中，目标是获得计算和存储收益。** 它把低位操作数交给支持相应格式的硬件计算路径，例如用低精度矩阵乘法处理线性层；可用于部署推理、rollout，也可用于训练的前向或反向。低位数据能减少相应张量的存储和搬运量，支持的硬件路径可能提高吞吐，实际收益还取决于转换开销与其他运算。[NVIDIA 的低精度推理说明](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/)

![原图：miniTE 的原生低精度训练路径 | 来源：miniTE v1（2026），Figure 2，CC BY 4.0。先看右侧 (b)：上方是 NVFP4 前向，下方是 BF16 反向；左侧是跨引擎检查，可留到排障时阅读。](https://arxiv.org/html/2610.23928v1/4_1_2.png)

**读图：** 右侧 (b) 的 Forward（前向）把输入和权重量化、打包为 NVFP4 数据，交给标有 **FP4 Tensor Cores** 的硬件矩阵乘法路径，再进行 FP32 缩放并输出 BF16 结果；下面的 Backward（反向）则使用保存的 BF16 数据计算梯度。因此，原生 FP4 前向可以与较高精度反向并存。图中是 miniTE 专家层的具体方案，不代表所有训练操作都降为 4 位。[miniTE v1 §4.2、附录 D](https://arxiv.org/html/2610.23928v1#S4.SS2)

两者也可以前后衔接：训练时用伪量化改善模型对低位数值的适应能力，部署时用原生低精度运算执行它。是否启用原生路径、是否变快，以及训练与 rollout 是否数值对齐，都需要分别验证。

**W4A4 只说明相应权重和激活的位宽。** W 是 Weights（权重），A 是 Activations（激活），两个 4 分别表示采用 4 位表示。使用这个标签还须注明适用层或算子；它既不区分 INT4（4 位整数）与 FP4，也不指定累加、梯度及优化器精度。训练与 rollout 都使用低精度，仍不代表两侧数值已经对齐（见 2.1、2.4）。

### 3.2 NVFP4 怎样表示一个数

**先读懂两种数值格式，再看缩放。** NVFP4 是 NVIDIA 的 4 位浮点格式方案：元素使用 E2M1，块尺度使用 E4M3。E 指指数（Exponent），M 指尾数（Mantissa）；E2M1 含 1 位符号、2 位指数、1 位尾数，E4M3 则含 1 位符号、4 位指数、3 位尾数。[NVIDIA 格式介绍，2025-06-24](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

下面用 $v$ 表示编码对应的数值，$S$ 表示符号位（0 为正、1 为负），$E$、$M$ 分别表示指数位、尾数位读出的非负整数。例如，二进制 `1000` 读作整数 8。**指数字段不是实际指数，要先减去该格式规定的“指数偏置”。** 正规数的有效数有一个不显式存储的首位 1；尾数位存的是它后面的小数部分。

**E2M1：元素怎样表示，最大值 6 从哪里来？** 指数偏置为 1，尾数只有 1 位，因此在 $E>0$ 时，$v=(-1)^S\times2^{E-1}\times(1+M/2)$，其中 $E\in\{1,2,3\}$、$M\in\{0,1\}$。最大正数的编码是 `0 | 11 | 1`（符号、指数、尾数），代入得到 $v=2^{3-1}\times(1+1/2)=6$。当 $E=0$ 时，使用零或非正规数规则 $v=(-1)^S\times M/2$，可表示 0 和 $\pm0.5$。合起来，非负可表示值就是 $0,0.5,1,1.5,2,3,4,6$，另有对应负值；它们是离散值，不能表示 $[-6,6]$ 中任意实数。[E2M1 编码：OCP MX v1.0，§5.3.3、表 5](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf#page=13)

**E4M3：尺度怎样表示，为什么能精确保存 2？** 指数偏置为 7，尾数有 3 位，因此有限正规数的公式是 $v=(-1)^S\times2^{E-7}\times(1+M/8)$；这里 $M/8$ 就是三位二进制小数。数值 2 的编码为 `0 | 1000 | 000`，即 $S=0,E=8,M=0$，所以 $v=2^{8-7}\times(1+0/8)=2$，没有舍入误差。上述公式适用于 $1\le E\le15$、$0\le M\le7$，但 $E=15,M=7$ 是保留的非数值编码；$E=0$ 时改用 $v=(-1)^S\times2^{-6}\times(M/8)$，表示零或非正规数。尺度也只能从 E4M3 的离散可表示值中选择，不是任意实数都能精确保存。[数值公式与 E4M3 编码：OCP MX v1.0，§5.3、表 2](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf#page=11)

**最大、最小值对照。** “最小值”指最负的有限数；“最小正数”指最接近零的正数。编码按 `符号 指数 尾数` 分组，以下都是尚未乘外部尺度的格式数值。

| 格式 | 边界 | 编码 | 代入计算 |
| --- | --- | --- | --- |
| E2M1 | 最大值 | `0 11 1` | $2^{3-1}(1+1/2)=6$ |
| E2M1 | 最小值 | `1 11 1` | $-2^{3-1}(1+1/2)=-6$ |
| E2M1 | 最小正正规数 | `0 01 0` | $2^{1-1}(1+0/2)=1$ |
| E2M1 | 最小正数（非正规） | `0 00 1` | $1/2=0.5$ |
| E4M3 | 最大值 | `0 1111 110` | $2^{15-7}(1+6/8)=448$ |
| E4M3 | 最小值 | `1 1111 110` | $-2^{15-7}(1+6/8)=-448$ |
| E4M3 | 最小正正规数 | `0 0001 000` | $2^{1-7}(1+0/8)=2^{-6}=0.015625$ |
| E4M3 | 最小正数（非正规） | `0 0000 001` | $2^{-6}(1/8)=2^{-9}=0.001953125$ |

E4M3 的最大值不能再把尾数从 `110` 增到 `111`：指数为 `1111`、尾数为 `111` 时表示 NaN（Not a Number，非数），不是有限数。**有限值范围分别为 $[-6,6]$ 和 $[-448,448]$，其中仍只有离散值可表示。** [OCP MX v1.0，表 2、表 5](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf#page=11)

**尺度负责缩放量级。** 原数据可能远大于 6，也可能全部很小，因此先除以尺度，映射到可表示范围并舍入，使用时再乘回尺度。基础的一维分块方案把每 16 个元素分为一块，并使用两级尺度：

| 记号 | 含义 |
| --- | --- |
| $x_i$、$b$ | $x_i$ 是待量化张量的第 $i$ 个实数，非第 1 章的 token ID；$b$ 是该元素所在块的索引集合。 |
| $c_i$ | 第 $i$ 个元素量化后的 E2M1 **数值**，不是表示它的原始位编码。 |
| $s_b$ | 块内 16 个元素共享的尺度，以 E4M3 保存。 |
| $g$ | 外层 FP32（32-bit floating point，32 位单精度浮点）尺度；基础说明中由整个张量共享。 |
| $\widehat{x}_i$ | 恢复后的近似值：$\widehat{x}_i=g\,s_b\,c_i$。 |

两级尺度结构见 [Transformer Engine NVFP4 文档](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html#data-format)。**下面先给定正的外层尺度 $g$，解释一个非零块怎样量化与恢复；不把它当作包含 $g$ 选取规则的完整算法。** $g$ 应由具体配置按其覆盖数据的范围确定；本文把它记作恢复时的乘数，代码中若存的是量化时的乘数，可能对应其倒数。

**手算一次：先算出块尺度 $s_b$。** 人为给定 $g=1$，某个 16 元素块的最大绝对值为 12，并采用最近值舍入。记 $Q_{\mathrm{E4M3}}$、$Q_{\mathrm{E2M1}}$ 为舍入到相应格式的可表示数值。

**① 计算并保存尺度。** 这里算的是**把元素缩放到 E2M1 范围所需的尺度**，不是 E4M3 的编码公式。元素缩放后是 $x_i/(g s_b)$；希望块内最大幅值落到 E2M1 的上限 6，就令 $\max_{i\in b}|x_i|/(g s_b)=6$，解得未舍入的尺度为 $\max_{i\in b}|x_i|/(6g)$。再用 E4M3 保存这个尺度，才得到 $s_b=Q_{\mathrm{E4M3}}\bigl(\max_{i\in b}|x_i|/(6g)\bigr)$。**分母的 6 由元素格式决定，外面的 $Q_{\mathrm{E4M3}}$ 由尺度存储格式决定；448 只是尺度格式的上限。**

代入本例的最大绝对值 12 和 $g=1$，得到 $s_b=Q_{\mathrm{E4M3}}\bigl(12/(6\times1)\bigr)=Q_{\mathrm{E4M3}}(2)=2$。前面已验证 2 的编码是 `0 | 1000 | 000`，因此本例的尺度恰好为 2，没有舍入误差；一般情况下，尺度舍入后，最大元素不一定恰好映射到 6。

**② 用算出的尺度量化元素。** 对块中 $x_i=5.2$，先除以 $g s_b=1\times2$，得到 $5.2/(1\times2)=2.6$。最近的 E2M1 候选值是 2 和 3，距离分别为 0.6 和 0.4，因此 $c_i=Q_{\mathrm{E2M1}}(2.6)=3$。这里用 E2M1 保存元素值 3，用 E4M3 保存共享尺度 2。

**③ 乘回尺度，恢复近似值。** $\widehat{x}_i=g s_b c_i=1\times2\times3=6$，比原值 5.2 大 0.8。**误差出现在 $2.6\to3$ 的元素舍入，不是尺度 2 的保存过程。** 这是教学设定，$g=1$ 不是默认配置，也不是 GPU 实验结果。

把这三步写成通用公式，就是下式：先计算 $s_b$，再用它计算 $c_i$，最后恢复 $\widehat{x}_i$。

**式 9：NVFP4 重建的示意**

$$
s_b=Q_{\mathrm{E4M3}}\!\left(\frac{\max_{i\in b}|x_i|}{6g}\right),\quad c_i=Q_{\mathrm{E2M1}}\!\left(\frac{x_i}{g s_b}\right),\quad \widehat{x}_i=g s_b c_i.
$$

这组表达式对应 [miniTE v1 §2、式 (3)](https://arxiv.org/html/2610.23928v1#S2)。全零块、尺度舍入为零或数值超出范围时，需要实现另行处理；不能直接用零尺度作除数。

**格式之外，还要指定 recipe（量化与计算配置）。** 它决定分块方式、尺度覆盖范围和舍入方式等。基础说明是 16 元素一块、外层按张量缩放；[Transformer Engine 的训练配置](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html#nvfp4-format) 还包括权重采用 $16\times16$ 共享尺度的变体，以及减轻量化偏差的随机舍入、缓和离群值影响的随机 Hadamard（哈达玛）变换。[miniTE v1 §4.2、附录 D](https://arxiv.org/html/2610.23928v1#A4) 则描述了外层尺度按行处理的具体路径。这些是不同配置，不能混成一种固定规则。

对**按行分块**的量化方案，矩阵转置后重新分块，可能让同一个元素与不同元素共享尺度，因此“先量化再转置”与“先转置再量化”通常不相等；这并非所有量化方案的必然性质。[Transformer Engine 的转置与二维分块说明](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)

本节引用的 Transformer Engine 在线文档在 2026-11-01 核对时标为 **2.20.2**；miniTE 使用固定 **v1**。这里依据文档解释结构与配置，未将在线文档视为第 3.6 节固定源码提交的行为保证。

### 3.3 为什么别的 token 会影响当前 token 的量化结果

**目标 token 没变，共享尺度却可能变。** 把当前层的激活排成矩阵，每一行是某个 token 的隐藏向量。`amax` 指参与统计的元素中**绝对值的最大值**，scale 是据此计算的缩放尺度。如果多行共用一个尺度，其他行出现大数，就可能改变目标行的缩放与舍入结果。这里的“邻居”是一起参与计算的其他 token 行，不一定在文本中相邻。

**先写规则，再手算。** 用一个对称 4 位整数（INT4，4-bit integer）教学量化器，选用整数范围 $[-7,7]$。记 $x$ 为单个激活元素、$s$ 为共享尺度、$\widehat{x}$ 为恢复值，规则是 $s=\operatorname{amax}/7$，$\widehat{x}=s\,\operatorname{clip}(\operatorname{round}(x/s),-7,7)$。分母 7 是此量化器的最大整数值；$\operatorname{round}$ 取最近整数，恰在中点时取偶数，$\operatorname{clip}$ 把结果限制在 $[-7,7]$。本例的最大绝对值非零，不涉及零尺度处理。

固定目标行 `[1, 0.5]`，只改变尺度的统计范围。共享尺度时，其他行含有数值 20，且所有元素的绝对值不超过 20：

| 情况 | 先算尺度 $s$ | 除以尺度，再舍入 | 乘回尺度，得到目标行恢复值 |
| --- | --- | --- | --- |
| 目标行独立尺度 | $\operatorname{amax}=1$，$s=1/7$ | $[1,0.5]/s=[7,3.5]$；舍入为 $[7,4]$ | $(1/7)[7,4]=[1,4/7]$，约为 $[1,0.5714]$ |
| 与其他行共享尺度 | $\operatorname{amax}=20$，$s=20/7$ | $[1,0.5]/s=[0.35,0.175]$；舍入为 $[0,0]$ | $(20/7)[0,0]=[0,0]$ |

第二种情况的尺度更大，同样的目标值除以尺度后更接近零，于是都被舍入成零。第一种情况中 $3.5\to4$ 用到了中点取偶数规则；两种情况都没有触发范围裁剪。**这是演示共享尺度影响的整数例子，不是 NVFP4 编码或真实模型实验。** 数字对应[本地教学脚本](../scripts/rl-infra-numerics-lab.py)的 `quantize` 函数与 `run` 中的 `single`、`mixed` 计算。

**per-token scale 改变的是“按哪些元素统计尺度”。** 每个 token 行独立确定尺度，称为 per-token（逐 token）或 per-row（逐行）缩放，不是每个元素单独一个尺度。接回 3.2 的两级结构，miniTE 的具体路径让每个激活行独立计算外层尺度 $g$，同时保留行内每 16 个元素共享的 E4M3 块尺度 $s_b$；因此逐 token 缩放不等于取消分块，也不等于整行只有一个尺度。[miniTE v1 §4.2](https://arxiv.org/html/2610.23928v1#S4.SS2)

**在目标行及其他量化规则固定时，按行独立算尺度，可以消除“其他行的最大绝对值改变目标行尺度”这条依赖。** 它不保证整个模型的输出不受批次影响，也不保证训练端与推理端一致：分块或随机舍入方式仍可能不同；即使量化值相同，进入不同的累加与计算路径，结果仍可能不同。miniTE 的特定对照也显示，仅改变推理端的尺度粒度不足以建立跨引擎一致性。[miniTE v1 §3.2、附录 C](https://arxiv.org/html/2610.23928v1#S3.SS2)

### 3.4 量化后难以求导，训练怎样更新参数

**为什么直接求导可能没有用？** 对固定尺度、固定量化网格、采用确定性舍入的量化函数，输入在一个小区间内变化时，量化结果可能完全不变，因此区间内的真实导数为零，跳变阈值处则不可导。如果尺度随输入变化、尺度可学习，或者采用随机舍入，就需要另行分析，不能直接套用这个结论。

**STE 在反向传播时指定替代导数。** Straight-Through Estimator（STE，直通估计器）的目的，是让训练获得可用于更新的信号；它没有把舍入函数变得可导，也不是在求其真实导数。本节先讲最简单的 identity STE（恒等直通估计器）：把量化节点的局部反向导数设为 1，让上游传来的梯度直接通过。早期来源 [Bengio 等（2013）](https://arxiv.org/abs/1308.3432)讨论的是随机神经元，不是这里的 NVFP4 实现。

这里 $w$ 是待更新的单个高精度权重，不是第 1 章的校正函数 $w(\rho)$；$Q(w)$ 表示量化再恢复到原单位的值，$\widetilde w$ 是实际送入后续计算的变量。沿用第 1 章的 $\operatorname{sg}$，即 stop-gradient（停止梯度）：前向保留数值，反向切断包住的计算路径。可用下面的表达式实现 identity STE；末项的 $\mathrm{STE}$ 下标与 $:=$ 明确表示**反向指定的规则**。

**式 10：前向量化，反向采用替代导数**

$$
\widetilde{w}=w+\operatorname{sg}(Q(w)-w),\qquad \widetilde{w}_{\mathrm{forward}}=Q(w),\qquad \left.\frac{\partial\widetilde{w}}{\partial w}\right|_{\mathrm{STE}}:=1.
$$

**前向怎么算？** $\operatorname{sg}$ 不改变数值，因此 $\widetilde w=w+(Q(w)-w)=Q(w)$。**反向走哪条路？** 括号内 $Q(w)-w$ 的路径停止求导，外面的 $w$ 路径仍保留，所以自动微分按 $1+0=1$ 传递梯度。**这个 1 只是局部规则，不是最终损失对权重的梯度；还要乘上游梯度。**

**手算一次从前向到更新。** 固定量化步长 0.25，令 $Q(w)=0.25\operatorname{round}(w/0.25)$、$w=0.43$；$\operatorname{round}$ 仍取最近整数，中点取偶数。损失取 $\mathcal L=\tfrac12(\widetilde w-1)^2$，目标值为 1：

- **前向：** $Q(0.43)=0.25\operatorname{round}(1.72)=0.5$，所以 $\widetilde w=0.5$，损失为 $\tfrac12(0.5-1)^2=0.125$。
- **真实前向的有限差分：** 只要 $0.375<w<0.625$，量化输出就仍是 0.5。取扰动 $\varepsilon=10^{-6}$，两侧损失都为 0.125，因此 $(0.125-0.125)/(2\varepsilon)=0$。
- **STE 反向：** 上游梯度是 $\partial\mathcal L/\partial\widetilde w=\widetilde w-1=-0.5$；乘量化节点指定的局部导数 1，传回权重的替代梯度就是 $(-0.5)\times1=-0.5$。
- **参数更新：** 若采用学习率 $\eta>0$ 的梯度下降，则 $w_{\mathrm{new}}=w-\eta(-0.5)=0.43+0.5\eta$。量化输出暂时没变，高精度权重仍可积累更新，之后跨过量化阈值。

这组数字来自[本地 CPU 教学脚本](../scripts/rl-infra-numerics-lab.py)的 `run` 中 `q`、`loss`、`fd`、`ste` 计算，不是 GPU 实验。其他 STE 可以在裁剪区间外把局部导数设为零；替代规则需要明确选择，不能承诺任意 STE 都能改善训练。[Yin 等（2019）](https://arxiv.org/abs/1903.05662)的理论分析有特定网络、数据分布与估计器条件。

**怎样验证实现？** 对原始量化前向做有限差分，通常不能验证 STE 人为指定的反向规则；这不代表实现可以免于测试。应分别检查前向量化结果，并将反向梯度与**明确采用同一替代规则**的参考实现比较。miniTE 附录 D 的梯度比较也明确采用直通估计器语义。[miniTE v1 附录 D、表 8](https://arxiv.org/html/2610.23928v1#A4)

这个标量例子只解释梯度怎样通过量化节点。实际矩阵层的反向还涉及保存原始操作数还是量化后的操作数；miniTE 的一种方案使用保存的原始 BF16 激活与权重，具体区别见 3.6，不能用上面的局部导数 1 概括整层反向。[miniTE v1 §4.2](https://arxiv.org/html/2610.23928v1#S4.SS2)

### 3.5 从报告提炼设计，不拼接不同实验的数字

| 阅读材料 | 本篇采用的关键结论 | 适用边界 |
| --- | --- | --- |
| [QaRL，2026-04，§3](https://arxiv.org/html/2604.07853v1) | 对齐低位训练前向与 rollout；反向保留高精度，并发布已形成的低位权重 | 仍允许残余差异及其校正，不等于普遍位级一致 |
| [miniTE v1，2026-10-07，§3–5](https://arxiv.org/html/2610.23928v1) | 同版本差异可在更新前出现；路由一致后仍有算子差异；实际学习路径与严格参考分开 | 默认 native W4A4 专家前向、BF16 backward，不能外推为全模型 FP4 |
| [vLLM，2026-10-14，§2–4](https://vllm.ai/blog/2026-10-14-train-inference-consistency) | 用推理数值路径定义训练 forward，并逐算子对齐 | 在文中验收模型/布局/任务上成立，非所有主线模型的开关保证 |
| [TRACE v1，§3 / 附录 A](https://arxiv.org/html/2610.07767v1) | 记录 rollout 量化信息，指导训练侧 rounding | 压缩 guidance 是局部参考；不是任意原始 FP4 codeword 的完整恢复 |

miniTE v1 在 Qwen3-30B-A3B 冻结评分中报告 BF16 与 NVFP4 的绝对 logprob 差 P99 分别为 0.1271 和 0.2911 nats；其严格执行参考与实用训练路径承担不同角色。论文的完整 learner 更新速度不能直接当成端到端 RL 加速。[miniTE §3、§5](https://arxiv.org/html/2610.23928v1)

检索还发现署名作者在 LMSYS **staging** 域名的 10 月 25 日后续稿，引入 TRIAGE，并单独讨论权重转换/同步抵消采样收益。这份可变稿件与 arXiv v1 的内容和实验口径不同，本文只将它列为后续阅读，不把两份材料拼成一个已复现配置。[后续稿与边界说明](https://staging.lmsys.org/blog/2026-10-25-minite)

vLLM 的上述文章报告特定 DeepSeek-V4-Flash 配置在 100 步中 sampled-token logprob diff 为零，rollout 吞吐为高性能基线的 91.3%；作者同时说明未完成长期端到端墙钟比较，基线调参有限。**“91.3% rollout 吞吐”既不是端到端训练加速比，也不是通用性能承诺。**[实验限定](https://vllm.ai/blog/2026-10-14-train-inference-consistency)

### 3.6 主线与实验 PR 的版本边界

核对的 Transformer Engine 固定提交已在 `NVFP4BlockScaling` 中提供 `row_scaled_activation`、`backward_override` 等配置；`high_precision` 保存原高精度操作数用于反向，`dequantized` 则从保存的量化操作数恢复到计算 dtype。这两种 backward 的输入不同，即使 forward 一样也不能混作同一实验。[固定源码](https://github.com/NVIDIA/TransformerEngine/blob/dfc4158d1597f72972fe2863adac436fa3fe1d4d/transformer_engine/common/recipe/__init__.py#L534)

以线性层 $Y=XW^\top$ 为例：$X$ 是输入激活矩阵，$W$ 是权重矩阵，$Y$ 是输出，$\top$ 表示转置。若 $X$ 的形状为 $M\times K$，$W$ 为 $N\times K$，则 $Y$ 为 $M\times N$；$M$ 是输入行数，$K,N$ 分别是输入和输出特征维度。上游梯度 $G=\partial\mathcal L/\partial Y$ 与 $Y$ 同形，常规反向为 $dX=GW$、$dW=G^\top X$，这里 $dX,dW$ 分别表示损失对输入和权重的梯度。如果反向使用保存的高精度 $X,W$，与使用反量化后的 $\widehat X,\widehat W$，得到的更新就可能不同。二者都可能采用高精度 GEMM，但操作数已经不同；“BF16 backward”这几个字还不足以复现实验。

[PR #3045](https://github.com/NVIDIA/TransformerEngine/pull/3045) 的作者说明其整体不准备按当前形式合并，功能通过小 PR 逐步进入主线。阅读其 per-token recipe 时，要同时记录 PR head、主线 commit 和实际安装版本。本文没有取得一个可验证的独立 miniTE 仓库固定提交，因此不虚构 miniTE 内部函数调用链；相关源码部分以可核对的 TE 实现与 PR 为边界。

## 第 4 章 MoE 路由与执行状态

### 4.1 微小数值差异为什么可能改变专家选择

**先看一个 token 怎样经过混合专家层。** Mixture of Experts（MoE，混合专家）让路由器为多个专家打分，选择部分专家计算，再把输出加权合并。这里的 top-k 是选择分数最高的 $k$ 个**专家**，不是第 1 章采样时选择候选 token。[R3 原论文 v1 §4.1](https://arxiv.org/html/2510.11370v1#S4.SS1)

**选择边界为什么敏感？** 把专家分数从高到低排列，排序后的第 $k$ 名是最后一个入选者，第 $k+1$ 名是第一个落选者；这里说的是名次，不是专家编号。两者分数越接近，越容易因微小扰动交换名次，使入选集合发生离散变化。这解释了为什么数值差异可能改变实际执行的专家。[R3 原论文 v1 §3.2](https://arxiv.org/html/2510.11370v1#S3.SS2)

**手算一个切换。** 假设采用 top-1，选中专家的合并权重为 1；两个专家的分数为 1.000、0.999，标量输出分别为 2、−2。起初选中第一个专家，输出为 $1\times2=2$；第二个分数增加 0.002 后成为 1.001，改选第二个专家，输出为 $1\times(-2)=-2$，输出差的绝对值为 $|-2-2|=4$。这是本文教学反例，说明小的分数扰动可以触发较大的输出变化，不代表真实模型中的典型误差量级。

用符号表示这一过程：本节 $x$ 是该层收到的一个 token 的隐藏向量；$u(x)$ 是路由分数向量，$I$ 是选中专家的编号集合，$E_i(x)$ 是专家 $i$ 的输出，$g_i(x)$ 是其合并权重（gate）。**分数决定选谁，权重决定其输出占多大比重**，两者不能直接等同；具体打分和归一化规则由模型决定。简化的稀疏专家分支输出为：

**式 11：离散专家集合与连续计算**

$$
F_I(x)=\sum_{i\in I}g_i(x)E_i(x),\qquad I=\operatorname{TopK}(u(x)).
$$

这里的 $u$ 对应 R3 论文的路由分数 $\mathbf{s}$，集合 $I$ 对应论文二值选择掩码中取 1 的位置；式号为本文编号。式 11 未包含共享专家等额外结构，不是所有 MoE 层的完整公式。[R3 原论文 v1 §4.1，式 (4)–(7)](https://arxiv.org/html/2510.11370v1#S4.SS1)

因此，排查时可以依次看：**入选边界的分数差多小、两侧选中的专家是否不同、切换时层输出差多少**。这是本文根据上述机制整理的观察顺序。分数接近不必然带来大输出差异，还取决于被替换专家的输出与合并权重。

### 4.2 路由重放如何固定专家选择，以及它的边界

**先记录选择，再按记录重算。** Rollout Routing Replay（R3，rollout 路由重放）在生成训练样本时，记录**每个 token 在每个 MoE 层**选中的专家；训练重算对应位置时，使用这份选择，继续计算当前模型的合并权重与专家输出。R3 论文具体保存并替换的是二值路由掩码（mask），其中 1 表示入选，0 表示未入选。[R3 原论文 v1 §4.1](https://arxiv.org/html/2510.11370v1#S4.SS1)

| 对象 | 重放时怎样处理 |
| --- | --- |
| 专家编号／选择掩码 | 使用 rollout 记录的选择 |
| gate 合并权重 | 根据当前路由分数和模型的归一化规则计算 |
| 专家输出 | 用当前输入和专家参数重新计算 |

**重放改变的是哪一次比较？** 沿用 $R$ 表示 rollout、$L$ 表示训练端，$x_R,x_L$ 是两侧进入该层的隐藏向量，$I_R,I_L$ 是各自选中的专家集合。把三个输出分别记为：$A=F^R_{I_R}(x_R)$ 是 rollout 输出，$B=F^L_{I_L}(x_L)$ 是训练端自行选择专家的输出，$C=F^L_{I_R}(x_L)$ 是训练端改用 rollout 选择后的输出。比较 $B$ 与 $C$ 时，固定该层的训练端输入、参数及其他计算配置，只替换专家选择；由加减同一个中间项 $C$，得到本文的代数分解：

**式 12：路由差异与同路由残差**

$$
A-B=(A-C)+(C-B).
$$

原始比较是 $A-B$；**重放后改为比较 $A-C$，并不是原先算出的 $C-B$ 自动变成零。** $C-B$ 描述在这次局部对照中替换专家选择的影响。全模型重放还可能改变后续层的输入，因此不能把这项分解当作整网误差的固定归因比例。

**选择相同，仍可能算出不同结果。** $A-C$ 仍可能包含两侧隐藏向量的差异、当前合并权重的差异，以及专家运算中的量化、累加和输出转换差异。miniTE 在目标层固定专家分配后仍观察到输出差异；其中保留自然输入的比较也包含上游差异，不能把残差全部归因于该层算术。要进一步隔离输入影响，需要按 2.4 的方法重放相同入口输入与相关状态。[miniTE v1 §3.2、附录 C](https://arxiv.org/html/2610.23928v1#A3)

**固定选择后，梯度怎样走？** 选择掩码使用记录值，但当前合并权重和专家计算仍可以参与反向传播；重放不等于跳过整个路由器或冻结相关参数。下面只考察“在选中集合内部做 softmax 归一化”的模型，与 R3 论文 §4.1 的写法对应：$u_i$ 是当前路由分数，$g_i$ 是合并权重，$\mathbf{1}_{i=j}$ 在 $i=j$ 时为 1，否则为 0。固定集合 $I_R$ 后，对当前分数求偏导，得到：

**式 13：固定集合内的 gate 导数**

$$
g_i=\frac{e^{u_i}}{\sum_{j\in I_R}e^{u_j}},\qquad \frac{\partial g_i}{\partial u_j}=g_i(\mathbf{1}_{i=j}-g_j),\quad i,j\in I_R.
$$

式 13 的导数是本文对上述 softmax 的数学推导：选中专家之间的相对分数仍可改变合并权重，但这不保证实际损失对路由器的梯度一定非零。**特别是 top-1：若只在唯一选中的专家内归一化，其权重 $g_i=e^{u_i}/e^{u_i}=1$，这条合并权重路径的导数就是 $1\times(1-1)=0$。** 其他归一化方式或辅助损失可能带来不同的梯度路径，不能套用这一特例。

**固定版本源码能确认到哪里？** slime 的重放分支读取记录的 `top_indices`，再用 `scores.gather(1, top_indices)` 提取**当前 scores**。这能确认复用了索引，没有直接复用旧的专家输出；仅凭这个包装函数，不能确定所有架构最终采用哪种 gate 归一化，也不能证明完整训练链已正确对齐。[slime 固定提交 `540ea91`：重放分支](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/utils/routing_replay.py#L180)

### 4.3 重放还有数据对齐与存储问题

路由信息必须与 trajectory、token 位置、MoE layer、top-k slot 一一对应。packing、CP 切片、padding 和重计算任何一处偏移，都可能“成功运行但重放了另一个 token”。应在小批次校验 ordered expert IDs，而不只比较集合；tie-break 与 combine 顺序也可能影响执行。

设 $T$ 是记录路由的 token 数，$n_{\mathrm{MoE}}$ 是 MoE 层数，$k$ 是每个 token 每层选择的专家数，$b_{\mathrm{ID}}$ 是每个专家 ID 占用的字节数。粗略存储量为 $T n_{\mathrm{MoE}} k b_{\mathrm{ID}}$ 字节：若 $T=32768$、$n_{\mathrm{MoE}}=40$、$k=8$、$b_{\mathrm{ID}}=2$，仅 IDs 就约 20 MiB/轨迹，还没有 mask、索引和协议开销。它是本文按假设计算的容量示例，实际 dtype/压缩方式以实现为准。

TRACE 将关注点进一步移到量化 rounding 的结果。其压缩记录为局部指导，论文伪代码会结合当前激活重建参考并选择相邻 FP4 候选。学习它时要问清楚“保存了哪些位、哪些 scale、哪些层”，以及这些信息的传输/存储代价。[TRACE §3 与附录 A](https://arxiv.org/html/2610.07767v1)

### 4.4 CUDA graph、权重更新与缓存必须一起考虑

PyTorch CUDA graph replay 使用捕获时的参数地址；可以更新这些地址里的内容，但简单把 Python 变量重新绑定到新 tensor，不意味着已捕获的 graph 会改读新地址。持久 buffer、原地复制和流同步是理解更新可见性的起点。[PyTorch CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)

本文建议的一次版本发布协议如下：先准备权重及 scales，再等待相关写入完成；在明确定义的边界切换可见版本；确认 graph/算子缓存引用正确；将 KV、recurrent state 和 prefix cache 绑定到生成它们的版本，必要时清空或重算；最后 ACK 并允许新请求读取。具体引擎可以使用不同协议，这里强调需要证明的约束。

如果旧权重算出的 KV 与新权重继续 decode 混用，即使所有 kernel 都确定，也不再对应“固定版本模型重算整个前缀”的分布。若系统有意采用跨版本续跑，应记录该执行语义；不能把样本笼统标成一个新版本，再声称实现了同版本对齐。

## 第 5 章 无 GPU 小实验：代码、结果与解释

### 5.1 如何运行与复查

[实验脚本](../scripts/rl-infra-numerics-lab.py) 只用 Python 标准库；不需要安装 PyTorch、CUDA 或下载模型。在站点源码根目录运行：

```sh
python3 scripts/rl-infra-numerics-lab.py
```

输出 [results.json](../assets/rl-infra-numerics/results.json) 和下方结果图。实验固定随机种子 `20261029`，包含浮点顺序、scale domain、STE、精确梯度与小 MoE 五组机制实验。脚本同时检查精确梯度与有限差分、IS 恒等式、版本差分解及 replay 后仍有非零残差。不同数学库的末位浮点结果可能略有差异。

所有结果是**本地 CPU 教学模型实测**：其中使用对称 INT4 式均匀量化和额外均匀 rounding，没有实现完整 NVFP4 编码、Tensor Core 或 NCCL。不能用它们报告 GPU 性能、大模型收敛或论文复现结果。

### 5.2 三个最小反例

| 实验 | 输入与控制变量 | 本地结果 | 说明 |
| --- | --- | --- | --- |
| FP32 加法顺序 | $10^8,-10^8,1$；每步强制 FP32 | 两种括号顺序得到 1 和 0 | 确定的两条路径也可不相等 |
| scale domain | 目标行 `[1,0.5]`，是否加入幅值 20 的邻居 | 独立行 `[1,0.571429]`；共用 scale `[0,0]` | 邻居可通过 amax 改变目标行 |
| STE | $w=0.43$、量化步长 0.25、目标 1 | 局部有限差分 0；identity STE -0.5 | STE 是替代梯度，不是真导数 |

### 5.3 IS 能恢复什么，截断又改变什么

固定一个前缀，使用 $p=(0.6,0.3,0.1)$、$q=(0.2,0.3,0.5)$，动作奖励 $(1,-0.5,0.2)$。三个向量的对应位置指向同一个候选动作；$p$ 是目标分布，$q$ 是行为分布。参数是 target softmax logits，baseline 为固定值 0.47。枚举三种动作得到精确目标梯度 $(0.318,-0.291,-0.027)$，并用中心有限差分验证。表中 L2 误差是估计梯度与目标梯度之差的欧氏范数，方向夹角比较两者的方向；$N$ 是样本数。

| 权重规则 | 对目标梯度的 L2 误差 | 方向夹角 | 理论 ESS/N |
| --- | ---: | ---: | ---: |
| 不校正 | 0.124064 | 15.502° | 1.000000 |
| 精确 $p/q$ | 约 $5.56\times10^{-17}$ | 0° | 0.471698 |
| TIS，上界 2 | 0.054050 | 1.351° | 0.571429 |
| 区间保留 $[0.5,2]$ | 0.177009 | 11.980° | 0.300000 |

这里枚举的是分布期望，没有 Monte Carlo 采样噪声。表中 ESS/N 是 $(\mathbb{E}_q w)^2/\mathbb{E}_q w^2$，不是有限 batch 的实测 ESS。这个例子展示截断的偏差—权重集中度取舍，不证明 TIS 总优于拒绝规则。实际 RL 的状态分布、奖励和优势误差还会改变结果。

### 5.4 小 MoE：自然执行、路由重放与版本变化

构造三个线性专家、二维输入、三个输出 token 的 top-1 MoE。为观察边界，故意让两个 router score 的差落在 $[-0.06,0.06]$，对 sampler 的第二个 score 增加 0.025；采样侧另施加输入和输出 rounding。生成 512 个前缀并按 $q$ 抽取一个 token。

三组对照：`natural` 用同权重 trainer 自己的路由；`route_replay` 只强制采用 sampler 路由，其余 trainer 算术不变；`version_shift` 在 natural 基础上把输出 logits 增加 `[0.2,-0.15,0.05]`，模拟明确的参数变化。三者使用完全相同的 sampled tokens。

![图 2 路由重放仍留下数值残差 | 本文 CPU 实验真实输出。绿色为平均绝对 logprob 差，金色为 P99；刻意构造的边界模型不代表真实 LLM 分布。](../assets/rl-infra-numerics/moe-results.svg)

| 对照 | 平均绝对 logprob 差 | P99 绝对差 | ratio 超出 $[0.8,1.2]$ |
| --- | ---: | ---: | ---: |
| natural | 0.151086 | 1.899141 | 15.04% |
| route_replay | 0.025383 | 0.080073 | 0.00% |
| version_shift | 0.236076 | 2.123448 | 25.78% |

natural 的路由翻转率为 17.1875%。replay 消除了本例的离散选择差异，但量化/算术残差仍在；越界率为零也不等于分布完全相同。版本变化进一步增加了此例的总差异，脚本逐 token 验证了式 1 中对应的加法分解。

本例没有训练 router，也没有测大模型的梯度夹角；第 5.3 节的夹角属于另一个精确三动作实验。不要把两个实验合成一个未经执行的“MoE 训练效果”。

### 5.5 小实验之后，GPU 上还要测什么

固定模型、硬件、数据和预算后，逐项增加变量：BF16 基线 → 仅低位权重 → 低位激活 → 低位 KV/state → routing replay → batch-invariant kernel → 跨 TP/EP → graph replay。每次记录实际改动，避免一次换完整框架后无法归因。

| 层级 | 最少需要的证据 | 判定边界 |
| --- | --- | --- |
| 量化算子 | codes、scale、反量化值、饱和率 | 固定输入与 recipe |
| 模块 | common-input 前向、surrogate backward 对照 | 区分原始/量化操作数反向 |
| 引擎 | 相同 token 的 logprob mean/P99/max、路由翻转 | 固定版本、前缀与采样语义 |
| RL 更新 | ratio 尾部、ESS、clip/reject、梯度范数与方向 | 相同有效 token/reduction |
| 学习质量 | 多 seed 的 reward、熵、长度、独立评测 | 固定总预算和数据 |
| 系统效率 | rollout、重算、训练、转换、传输、等待各阶段 | 单独报告端到端墙钟 |

## 第 6 章 固定版本源码走读与阅读材料

### 6.1 版本清单与复查方式

以下是本次实际取得的源码提交，不代表 release 标签。完整路径与 SHA-256 在[来源清单](../assets/rl-infra-numerics/sources.json)。复制参考代码时还需遵守各仓库许可；本文没有复制整段第三方实现。

| 仓库 | 固定提交 | 本次阅读范围 |
| --- | --- | --- |
| slime | `540ea919e45b8b929198abce008a103be9cfc779` | loss、ppo_utils、actor、routing_replay、sglang_rollout、policy-mismatch 文档 |
| vLLM | `61a6f8138fd81f89467af762dc1b74674f544300` | V1 sampler 的概率语义 |
| SGLang | `f6649627836699f76a3054d0812c05ac332932c0` | sampler 的 logprob/temperature 分支 |
| Transformer Engine | `dfc4158d1597f72972fe2863adac436fa3fe1d4d` | NVFP4 recipe 与 backward 配置 |

### 6.2 沿 slime 找到一次 loss 的三种概率

先沿样本确认 prompt/response 边界。[`sglang_rollout.py` 的初始化](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/rollout/sglang_rollout.py#L208) 用输入 token 建立样本，再[追加生成的 response tokens](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/rollout/sglang_rollout.py#L240)。训练侧 [`get_responses`，L124–150](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/backends/megatron_utils/loss.py#L124) 接收 prompt+response，在无 context parallel 的分支取 `tokens[-response_length:]` 为目标 token，并与前移一位的 `logits[start - 1 : end - 1]` 对齐。在因果模型下，这正是用 $h_t$ 预测 $a_t$。

实现还支持同一回答中断后的续生成：[`_prepare_prompt_ids`，L43–62](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/rollout/sglang_rollout.py#L43) 的复用分支直接返回已有 `sample.tokens`。因此该分支的 `prompt_ids` 是请求开始时的完整前缀，可能已含部分回答；读取代码时要按数据内容映射到 $h_t$，不能仅凭变量名把它映射到固定 $x$。续生成没有改变第 1 章的 prompt/response 切分。这些是固定版本的静态源码观察，未运行 GPU 框架。

先定位 [actor.py 的重算与 replay 分支](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/backends/megatron_utils/actor.py#L487)：普通路径在优化前重算 actor logprob，部分配置允许复用 loss forward；rollout replay 路径分别安排 forward 和 backward/recompute 的记录消费。阅读时先把“哪一次 forward”写在纸上，再跟随数据进入 loss。

| 入口 | 必须追问的问题 |
| --- | --- |
| [policy_loss_function，L1128](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/backends/megatron_utils/loss.py#L1128) | `use_rollout_logprobs` 是否使 old 指向 rollout？当前概率在哪里生成？ |
| [old/TIS 来源，L1161–1185](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/backends/megatron_utils/loss.py#L1161) | 缺少独立重算时是否使用当前值的 detached 副本？REINFORCE 分支与 PPO 有何区别？ |
| [vanilla_tis_function，L904](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/backends/megatron_utils/loss.py#L904) | 分子来自哪份 train logprob？clip 作用在权重还是 PPO ratio？ |
| [importance_weights，L25](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/utils/ppo_utils.py#L25) | clamp 与区间置零有何不同？实参是否覆盖默认阈值？ |
| [compute_policy_loss，L297](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/utils/ppo_utils.py#L297) | 最小化负号如何把 min 变成 max？dual-clip 是否启用？ |

这一固定版本中的 `ppo_kl` 在普通 token 分支是 old logprob 减 current logprob；变量名不意味着它就是全词表 KL。TIS 回调之后还有 mask/reducer 处理，主路径保留原 rollout normalizer。自定义回调或 reducer 可改变语义，读完一个函数不能替代检查完整配置。以上是源码观察，不是所有 slime 版本的契约。

建议自己画出 `[每样本 response tokens] → 拼接 → token loss → mask → reduction` 的 shape 表；再分别写“有/无独立 old 重算”“有/无 rollout logprob”“PPO/REINFORCE”四组配置，检查每个分母真正对应哪个分布。

### 6.3 推理端与量化端的阅读任务

对 [vLLM sampler](https://github.com/vllm-project/vllm/blob/61a6f8138fd81f89467af762dc1b74674f544300/vllm/v1/sample/sampler.py)，标出 raw logprob、logits processors、temperature、top-k/top-p 与返回 logprob 的先后关系。若继续到下游 sampler，必须固定同一个 commit，不能混读另一个版本。

对 [SGLang sampler](https://github.com/sgl-project/sglang/blob/f6649627836699f76a3054d0812c05ac332932c0/python/sglang/srt/layers/sampler.py#L231)，比较返回原始概率与对齐路径；特别注意 `log(softmax(x))` 和 `log_softmax(x)` 数学等价却可能数值不同，cast 与 temperature 的先后也属于执行定义。

对 [TE recipe](https://github.com/NVIDIA/TransformerEngine/blob/dfc4158d1597f72972fe2863adac436fa3fe1d4d/transformer_engine/common/recipe/__init__.py#L534)，列一张精度表：forward operand、accumulator、epilogue、saved tensor、dgrad、wgrad、master weight、optimizer。不能只填写“NVFP4”一格就结束。

### 6.4 推荐阅读顺序与证据等级

符号对照：[slime 的固定版本说明](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/docs/en/advanced/policy-mismatch.md#L9) 使用本文的 $x,y,h_t$；[DeepSeekMath §4.1.1，式 (1)](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS1) 使用 $\pi_\theta(o_t\mid q,o_{<t})$。后者的问题 $q$ 对应本文的 $x$，输出 $o$ 对应 $y$，输出 token $o_t$ 对应 $a_t$。该论文的 $q$ 是问题，不是本文的行为分布。读论文时先做这张记号映射，再比较公式。

1. 先读 [slime 三种概率与校正](https://thudm.github.io/slime/advanced/policy-mismatch.html)，再按第 6.2 节核对当前代码分支。
2. 读 [PyTorch 数值说明](https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html) 和 [Thinking Machines 的 batch invariance 分析](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)，运行浮点顺序反例。
3. 读 [R3 原论文](https://arxiv.org/html/2510.11370v1)、[QaRL](https://arxiv.org/html/2604.07853v1)，区分重放选择、对齐量化与残余校正。
4. 精读 [miniTE v1 的诊断与实验](https://arxiv.org/html/2610.23928v1)、[TE PR #3045](https://github.com/NVIDIA/TransformerEngine/pull/3045) 和 [vLLM 一致性报告](https://vllm.ai/blog/2026-10-14-train-inference-consistency)，逐个记录实验计时范围。
5. 最后读 [TRACE](https://arxiv.org/html/2610.07767v1) 与 [miniTE 后续 staging 稿](https://staging.lmsys.org/blog/2026-10-25-minite)，整理还未解决的问题，而非将更新的标题视为最终答案。

本文核心结论依据原论文、官方文档与固定源码。论文中的性能是作者报告；源码观察只确认该提交的实现；第 5 章数字来自本地脚本。后续复现时应连同模型 checkpoint、依赖、驱动和 kernel 版本一起固定。

## 第 7 章 排障流程、验收与下一步

### 7.1 遇到 logprob mismatch，按什么顺序查

![图 3 从语义到算子的排障顺序 | 本文原创排障流程。先固定比较对象，再定位第一个数值分叉，最后讨论校正和性能。](../assets/rl-infra-numerics/diagnosis.svg)

先核对 token/mask/shift 与概率语义，再核对 weight version、权重/scales hash 和 cache provenance。若同引擎重复就不同，优先解决 nondeterminism/状态污染；若换邻居才不同，查 scale domain 与 batch-dependent kernel；若 prefill/decode 不同，查 cache 精度、位置与增量算子；跨引擎差异最后逐层二分定位。确认来源以后，再选择算法校正、QAT、route replay 或共享数值执行路径。

这是按排障成本组织的建议顺序，不是所有系统唯一正确的诊断流程。运行中若发现明确的错误版本或 token 对齐错误，应先修复语义问题，不能靠扩大 TIS 阈值掩盖。

### 7.2 哪个指标回答哪个问题

下面 $\delta_t$ 是同一已采样 token 上的有符号 logprob 差；$L,U$ 是人为设定的 ratio 下、上阈值，满足 $0<L\le U$，不是引擎标签。$c_t$ 是越界指示量，越界取 1，否则取 0；$\lor$ 表示“或”。它只记录是否越界，是否据此裁剪或丢弃样本由具体算法决定。这里的 $c_t$ 与第 3 章的量化值 $c_i$ 无关。P95/P99 分别表示统计量的第 95/99 百分位数。

**式 14：用于日志的三组量**

$$
\delta_t=\log p_0(a_t\mid h_t)-\log q(a_t\mid h_t),\qquad \rho_t=e^{\delta_t},\qquad c_t=\mathbf{1}_{\rho_t<L\ \lor\ \rho_t>U}.
$$

报 mean/P95/P99/max 的 $|\delta|$，保留 signed gap、长度分桶、路由翻转条件下的统计，再联合 $A$ 与更新方向看高风险 token。若声明严格位级一致，就用精确相等的判据，并明确比较范围；若做近似对齐，则预先设阈值并验证它与训练风险的联系。

只在一批 sampled tokens 上测到零差异，是该测试集上的验收结果，不是对所有前缀、全词表和全部执行配置的证明。更强的保证还需要明确的算子约束、支持范围和覆盖测试；随机量化与采样也必须记录其随机状态及比较条件。

不要将 sampled-token 的绝对 logprob 差称作 KL。若 token 确实来自 $q$，固定前缀下 $\mathbb{E}_q[\log q-\log p_0]$ 才对应 $D_{KL}(q\Vert p_0)$；有限样本均值可为负。若返回的是 raw logprob，却按 processed 分布采样，连这个估计解释也需要重新检查。

只追平均值还不够：少数高权重、长轨迹或大优势 token 可能支配更新。恢复数据、分桶与尾部图，比只盯一条全局 KL 曲线更有诊断价值。

### 7.3 七个验收问题与回答要点

| 问题 | 应能说清的要点 |
| --- | --- |
| 同 checkpoint 为什么不是同 policy？ | 执行 recipe、kernel、cache 与采样变换也定义分布 |
| PPO ratio=1，为什么仍需行为校正？ | 它只比较 current 与 old trainer；行为分布另有来源 |
| TIS 能完全修复异步偏差吗？ | 支持集、前缀分布、截断偏差与轨迹方差均有限制 |
| R3 后 mismatch 为什么没归零？ | expert IDs 相同不保证输入、gate、量化和 GEMM 相同 |
| 为什么 STE 不通过普通有限差分？ | 它是定义的替代梯度，需对照 surrogate backward |
| 开启 batch invariant 就能与 trainer 对齐吗？ | 还要覆盖 phase、并行布局和跨引擎数值语义 |
| 如何证明一次低精度优化有效？ | 分别验证数值、梯度、学习质量、模块性能与端到端成本 |

### 7.4 完成项与待验证边界

本篇已完成概念推导、固定版本源码入口、五组 CPU 实验及对应结果解释。下一步最值得做的是选一份小型 dense 模型，在相同权重与固定 token 上建立 trainer/sampler logprob 对照，再逐项加入量化、MoE、cache、并行和异步因素。

真实 GPU 上的 batch/phase/TP 一致性、低位 kernel backward、graph 更新可见性、多 seed RL 收敛与端到端时间，仍属于待实测项目。它们不是本次 CPU 实验能够替代的结论。

[返回笔记目录](../)
