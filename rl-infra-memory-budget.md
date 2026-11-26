# MoE 训练显存估算

用 **Qwen3-30B-A3B 的 8 卡训练配置**手算三笔单卡显存：权重、梯度和 AdamW 状态占多少，为反传保存的层输入与 LM head 的临时张量占多少，EP 和 TP 通信又要留多少缓冲。第 5 章用同一模型估算 SGLang 的 rollout 显存和通信。所有结果均为按张量形状计算的估值。

[返回 Megatron 分布式训练与恢复](./rl-infra-megatron.html) · [RL Infra 学习路线](./rl-infra-roadmap.html)

- **模型**有 48 层，隐藏维度 2,048，词表 151,936。每层 128 个路由专家，每个 token 选 8 个。参数量以模型配置和权重形状为准。 [Qwen3 官方配置](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json) · [专家矩阵形状](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L192-L222)
- **并行配置**为 `d2t4 | d1e8`。普通参数走 DP=2、TP=4，路由专家走专家 DP=1、EP=8、ETP=1。PP=1、CP=1，开启 SP。两套分组各自覆盖同一批 8 张卡。 [Megatron MoE Parallel Folding](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html#moe-parallel-folding)
- **训练负载**按每个全局 microbatch 含 2 条序列估算，每条 32,768 token。除以普通 DP=2，每个副本同时处理 1 条。采用 BF16、FP32 AdamW 状态、Megatron 分布式优化器、FlashAttention 和逐层重计算。
- **符号与单位**。$B_\mu$ 是全局一个 microbatch 的样本数，$S$ 是每条序列的 token 数，$H$ 是隐藏维度，$V$ 是词表大小，$K$ 是每个 token 选的专家数。1 MiB = $2^{20}$ 字节，1 GiB = $2^{30}$ 字节。

如果手中是一次优化器更新的全局 batch，先除以梯度累积步数得到全局 microbatch，再除以普通 DP 得到每个副本处理的样本数。 [Megatron batch 定义](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/training-loop-settings.html)

## 第 1 章 静态训练状态

**先按 PP 划分当前 stage 持有的参数，再对这些参数做 TP 或 EP/ETP 分片。** 本例 PP=1，唯一的 stage 持有全模型，因此后面的数字直接用全模型参数量。有多个 PP stage 时，分别算每个 stage 的本地状态，按占用最高的 stage 做静态预算。 [Megatron PP stage 划分](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/pipeline_parallel_layout.html)

### 1.1 每个本地参数占多少字节

本地参数量指做过 TP、EP、ETP 切分后，当前 rank 实际持有的参数量。训练状态显存等于**本卡参数量 × 每个本地参数在本卡占的字节数**。

**式 1 · 每个本地参数的训练状态**

$$
\underbrace{2}_{\text{BF16 权重}}+\underbrace{4}_{\text{FP32 梯度缓冲}}+\frac{\underbrace{4}_{\text{FP32 主权重}}+\underbrace{4}_{\text{AdamW }m}+\underbrace{4}_{\text{AdamW }v}}{\underbrace{D}_{\text{对应 DP 组大小}}}=6+\frac{12}{D}
$$

BF16 模型权重和 FP32 梯度缓冲合计 **6 字节／参数**，按本卡分片完整分配。反传先在本地梯度缓冲上积累，再做 DP Reduce-Scatter。FP32 主权重和两个 AdamW 状态合计 **12 字节／参数**，由该参数对应的 DP 组平分。普通参数代入 $D=2$，每参数 **12 字节**。专家代入 $D=1$，每参数 **18 字节**。 [Megatron 分布式优化器](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html)

### 1.2 路由专家

28.991B 是全模型路由专家参数量。全模型参数量按 48 个 MoE 层 × 每层 128 个专家 × 每专家 3 个投影（gate、up、down）× 每个投影 2,048 × 768 个参数计算。EP=8 分专家，ETP=1 表示单个专家的权重矩阵由所属 rank 完整持有。对 stage $s$，本卡路由专家参数量是 $P^{(s)}_{\mathrm{expert}}/(\mathrm{EP}\times\mathrm{ETP})$。其中 $P^{(s)}_{\mathrm{expert}}$ 只统计该 stage 上的专家参数。

**式 2 · 路由专家训练状态**

$$
\underbrace{\frac{28.991\cdot10^9}{8\times1}}_{\text{专家参数量}\,\div\,\text{EP 8}\,\div\,\text{ETP 1}}\times\underbrace{\left(6+\frac{12}{1}\right)}_{\text{专家 DP 1，每参数 18 字节}}\;\approx\;60.75\ \mathrm{GiB}
$$

### 1.3 其余参数

对 stage $s$，本卡非路由专家参数量是 $P^{(s)}_{\mathrm{shard}}/\mathrm{TP}+P^{(s)}_{\mathrm{replica}}$。第一项把该 stage 上可切分的参数按 TP 均分，第二项完整计入 TP 复制的参数。若各 stage 的非专家参数量接近，且大部分参数都走 TP 切分，可以粗估为 **全模型非专家参数量 ÷ (TP × PP)**。精算时把 embedding、LM head 放到各自实际所在的 stage，再分别计算 router 和 norm 等复制参数。

PP=1 的这个算例中，可按 TP 切分的部分约 1.52830B，主要是 attention 投影、embedding 和 LM head。按复制估算的 router 和 norm 约 0.01279B，完整记入每卡。B 表示十亿个参数。实际装载时，词表 padding 和少量参数的 dtype 会使结果略有变化。

**式 3 · 其余参数训练状态**

$$
\left(\underbrace{\frac{1.52830\cdot10^9}{4\times1}}_{\text{全模型可切分参数量}\,\div\,(\text{TP 4}\times\text{PP 1})}+\underbrace{0.01279\cdot10^9}_{\text{本卡复制参数量}}\right)\times\underbrace{\left(6+\frac{12}{2}\right)}_{\text{普通 DP 2，每参数 12 字节}}\;\approx\;4.41\ \mathrm{GiB}
$$

**静态训练状态合计约 65.16 GiB／卡**，包括 60.75 GiB 专家状态和 4.41 GiB 其余训练状态。

## 第 2 章 激活与 LM head

### 2.1 为反传保存的层输入

逐层重计算按**每层保存一份 BF16 层输入**估算。$L$ 是模型总层数。各 PP stage 均分层数时，本 stage 保存的输入份数按 $L/\mathrm{PP}$ 算。分层不均时，代入该 stage 实际保存输入的层数。对这些沿序列维保存的输入，开启 SP 时 $\mathrm{TP}_{\mathrm{SP}}=\mathrm{TP}$，关闭时取 1。

**式 4 · 单卡保存的层输入**

$$
\underbrace{\frac{B_\mu}{\mathrm{DP}}}_{\text{本副本的序列数}}\times\underbrace{\frac{L}{\mathrm{PP}}}_{\text{模型总层数}\,\div\,\text{PP stage 数}}\times\underbrace{\frac{S}{\mathrm{CP}\,\mathrm{TP}_{\mathrm{SP}}}}_{\text{每条序列本卡保存的 token}}\times\underbrace{H}_{\text{每 token 维度}}\times\underbrace{2}_{\text{BF16 字节}}
$$

代入 **1 条序列／DP 副本 ×（模型 48 层 ÷ PP 1）×（32,768 token ÷ CP 1 ÷ TP 4）× 2,048 个隐藏维度 × 2 字节（BF16）**，得到 **1.50 GiB／卡**。重计算当前层时的中间张量按当时的工作区计入峰值。 [Megatron 激活重计算](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html) · [Megatron CP 与 SP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)

### 2.2 LM head 一次生成整个 microbatch 的 logits

LM head 的权重已经计入静态状态。下面算的是输出 stage 一次物化的 **BF16 logits**。LM head 聚合了所需的 SP 序列片，TP 在输出上切分词表。公式按保留本地 CP token 和 TP 词表分片的路径估算。

**式 5 · 全量 BF16 logits**

$$
\underbrace{\frac{B_\mu}{\mathrm{DP}}}_{\text{本副本的序列数}}\times\underbrace{\frac{S}{\mathrm{CP}}}_{\text{每条序列本卡处理的 token}}\times\underbrace{\frac{V}{\mathrm{TP}}}_{\text{本卡负责的词表大小}}\times\underbrace{2}_{\text{BF16 字节}}
$$

代入 **1 条序列／DP 副本 ×（32,768 token ÷ CP 1）×（151,936 个词 ÷ TP 4）× 2 字节（BF16）**，得到 **2.32 GiB／卡**。同形状的 FP32 logits 按每元素 4 字节计，一份是 **4.64 GiB／卡**。LM head 在输出端按一次计。每个副本若在同一个 microbatch 处理两条等长序列，logits 大小就变成 **4.64 GiB／卡**。

### 2.3 LM head 按 token 分块

分块路径把本地 token 展平，再按 $C$ 个 token 分块。$b_{\mathrm{logit}}$ 取实际 dtype 的字节数，例如 FP32 为 4。$B_\mu/\mathrm{DP}$ 影响总 token 数，单次缓冲按当块的实际 token 数算。

**式 6 · 单块 logits**

$$
\underbrace{\min\left(C,\frac{B_\mu}{\mathrm{DP}}\frac{S}{\mathrm{CP}}\right)}_{\text{本次分块实际处理的 token}}\times\underbrace{\frac{V}{\mathrm{TP}}}_{\text{本卡负责的词表大小}}\times\underbrace{b_{\mathrm{logit}}}_{\text{每个 logit 的字节数}}
$$

块上限取 **2,048 token** 时，单块 FP32 logits 为 **2,048 个 token ×（151,936 个词 ÷ TP 4）× 4 字节 ≈ 0.29 GiB／卡**。本例的块大小与模型隐藏维度恰好同为 2,048，二者分别对应不同的轴。 [AReaL 分块 LM head 源码](https://github.com/areal-project/AReaL/blob/main/areal/models/mcore/vocab_parallel_head.py#L548-L656)

## 第 3 章 通信缓冲与传输量

通信有两项预算。**缓冲显存**看当前 rank 同时需要存放多少字节。**单向传输量**看一个 rank 经通信发给其他 rank 多少字节。训练步的总传输量再按每个算子的实际调用次数累计。

### 3.1 EP All-to-All 的 token dispatch

**发送条目数**由本卡分到的 token 乘 Top-K 得到。这里按 ETP=1、无容量 padding、每条路由记录携带一份 hidden state 估算。

**式 7 · 本卡的 EP 发送条目数**

$$
N_{\mathrm{send}}=\underbrace{\frac{B_\mu}{\mathrm{DP}}}_{\text{本副本的序列数}}\times\underbrace{\frac{S}{\mathrm{CP}\,\mathrm{TP}_{\mathrm{SP}}}}_{\text{每条序列在本卡的 token}}\times\underbrace{K}_{\text{每 token 选几个专家}}
$$

**平均接收条目数**等于 EP 组内不同 token 的总数 × Top-K ÷ EP。当前 EP 组覆盖两个普通 DP 副本的全部序列分片，所以式中的 $D_g=2$。换并行配置时，按新的 EP 组实际覆盖的 token 数计算。

**式 8 · 本卡的 EP 平均接收条目数**

$$
N_{\mathrm{recv,avg}}=\underbrace{\frac{B_\mu}{\mathrm{DP}}}_{\text{每副本的序列数}}\times\underbrace{D_g}_{\text{EP 组覆盖的普通 DP 副本数}}\times\underbrace{S}_{\text{每条序列的 token}}\times\frac{\underbrace{K}_{\text{每 token 选几个专家}}}{\underbrace{\mathrm{EP}}_{\text{几个 EP rank 均分}}}
$$

代入 **本副本 1 条序列 ×（32,768 token ÷ CP 1 ÷ TP 4）× Top-K 8**，本卡发 65,536 条。代入 **1 条序列／副本 × 2 个副本 × 32,768 token × Top-K 8 ÷ EP 8**，本卡平均收 65,536 条。

**式 9 · EP 收发缓冲**

$$
\underbrace{(N_{\mathrm{send}}+N_{\mathrm{recv}})}_{\text{同时存放的发送和接收条目数}}\times\underbrace{H}_{\text{每条记录携带的隐藏维度}}\times\underbrace{2}_{\text{每个 BF16 元素的字节数}}
$$

每份缓冲是 **65,536 条 × 2,048 个隐藏维度 × 2 字节（BF16）= 256 MiB**。两份同时存在时，单次 dispatch 约 **0.50 GiB／卡**。估峰值时用最忙 EP rank 的实际接收条目数。算单向跨 rank 传输量时，用实际发往其他 rank 的记录数 × 隐藏维度 × dtype 字节数。整步再分别累计 dispatch、combine 及反向对应的通信。 [Megatron EP All-to-All 实现](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py#L645-L680)

### 3.2 TP 序列 All-Gather

输入在 TP 组内沿序列维分片。输出已拼齐整个 TP 组的序列片，所以输出缓冲里每条序列保留 $S/\mathrm{CP}$ 个 token。

**式 10 · TP 聚合后的输出缓冲**

$$
\underbrace{\frac{B_\mu}{\mathrm{DP}}}_{\text{本副本的序列数}}\times\underbrace{\frac{S}{\mathrm{CP}}}_{\text{每条序列聚合后的 token}}\times\underbrace{H}_{\text{每 token 维度}}\times\underbrace{2}_{\text{每个 BF16 元素的字节数}}
$$

代入 **1 条序列／副本 ×（32,768 token ÷ CP 1）× 2,048 个隐藏维度 × 2 字节（BF16）**，得到 **128 MiB／卡**。原输入分片的 32 MiB 按其用途计入激活。环形 All-Gather 中，每 rank 单向发出 **32 MiB 本地分片 ×（TP 4−1 次环形交换）= 96 MiB**。TP 的其他 All-Gather 和 Reduce-Scatter 按实际张量形状和调用次数另算。 [NCCL All-Gather 环形系数](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allgather) [Megatron TP 与序列并行](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html#sequence-parallelism-sp)

### 3.3 DP / 专家 DP 的梯度与参数同步

这里按 Megatron **分布式优化器**估算。反传对 FP32 梯度做 Reduce-Scatter，更新后对 BF16 权重做 All-Gather。$P_{\mathrm{local}}$ 是本 rank 持有的这组参数的元素数，已做 TP、PP 或 EP、ETP 切分。$D_c$ 取实际参与 collective 的组大小。普通和专家参数用各自的组，运行时若把 CP 并入普通梯度同步组，也按实际组大小代入。

环形 Reduce-Scatter 或 All-Gather 各执行一次时，下式估算**每 rank 单向发出的字节数**。接收方向同量。$b$ 在这里对 FP32 梯度取 4 字节，对 BF16 权重取 2 字节。

**式 11 · DP 单次环形 collective 的单向传输量**

$$
\underbrace{P_{\mathrm{local}}}_{\text{本卡 DP 同步的参数元素数}}\times\underbrace{b}_{\text{每个梯度或权重元素的字节数}}\times\underbrace{\frac{D_c-1}{D_c}}_{\text{通信组 }D_c\text{ 个 rank 的环形系数}}
$$

本例普通 DP=2，约 0.395B 本卡参数。**梯度约 0.74 GiB，参数约 0.37 GiB，合计约 1.10 GiB／卡／优化器步**，按各参数完成一次梯度同步和一次权重同步计。专家 DP=1，专家参数的 DP 同步流量为 **0**。这两项 collective 复用的参数和梯度大缓冲已经计入静态状态。若实现额外分配 staging 或临时缓冲，只取峰值时还存活的新增分配。 [Megatron 分布式优化器](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html) · [NCCL 环形通信带宽系数](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)

### 3.4 CP 的 Attention KV 交换

CP 把一条序列分给多个 rank。Attention 需要其他 CP rank 的 K、V，下面算**一份本地 KV 分片**有多大。$n_{\mathrm{KV,local}}$ 用当前 attention rank 实际持有的 KV head 数，已经考虑 TP 切分或复制。$d_{\mathrm{head}}$ 是每个 KV head 的宽度。BF16 时 $b_{\mathrm{KV}}=2$。

**式 12 · CP 的一份本地 KV 分片**

$$
\underbrace{\frac{B_\mu}{\mathrm{DP}}}_{\text{本副本的序列数}}\times\underbrace{\frac{S}{\mathrm{CP}}}_{\text{一个 CP 分片的 token 数}}\times\underbrace{n_{\mathrm{KV,local}}\,d_{\mathrm{head}}}_{\text{本卡 KV head 数}\times\text{每 head 宽度}}\times\underbrace{2}_{\text{K 和 V 各一份}}\times\underbrace{b_{\mathrm{KV}}}_{\text{每个元素的字节数}}
$$

对环形 P2P 的前向路径，**每 rank 发出的字节数约为一份 KV 分片 ×（CP−1 次交换）**，按每层 attention、每个 microbatch 计。反向再按实现计入 KV 交换和 KV 梯度归约。估显存时，按实际同时占用的接收块、双缓冲或 gathered KV 大小计算。该算例 **CP=1，跨 CP rank 流量和接收缓冲为 0**。作为换配置的示例，若 CP=2、TP=4，仍是每副本一条 32K 序列，**16,384 token × 本卡 1 个 KV head × 128 维 × K/V 两份 × BF16 2 字节 = 8 MiB／块**。 [Megatron 的 CP 前向与反向通信](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)

### 3.5 PP 的相邻 stage 点对点通信

**前向发层输出，反向发对应的激活梯度。** 按普通 decoder、边界张量沿序列维进行 SP 分片的 Megatron 路径，下面算**一次跨 stage 消息的字节数**。$H_{\mathrm{PP}}$ 是过流水线边界的每 token 维度，本模型取 2,048。$b_{\mathrm{PP}}$ 用实际流水线传输 dtype，BF16 时取 2。

**式 13 · PP 相邻 stage 的一次激活或梯度消息**

$$
\underbrace{\frac{B_\mu}{\mathrm{DP}}}_{\text{本副本的序列数}}\times\underbrace{\frac{S}{\mathrm{CP}\,\mathrm{TP}_{\mathrm{SP}}}}_{\text{每条序列在本卡的 token}}\times\underbrace{H_{\mathrm{PP}}}_{\text{跨 stage 的每 token 维度}}\times\underbrace{b_{\mathrm{PP}}}_{\text{每个传输元素的字节数}}
$$

对一条 stage 边界，按实际经过的 microbatch 数分别累计前向和反向消息。估显存时，看 1F1B 等调度下同一时刻还在的接收缓冲与激活。这个算例 **PP=1，没有相邻 stage，跨 stage 流量和接收缓冲为 0**。若增设 PP stage 且仍用 CP=1、TP=4 SP、本地 1 条 32K 序列，**一条边界单次 BF16 消息为 8,192 token × 2,048 维 × 2 字节 = 32 MiB／rank**。这里的 PP 大小决定边界个数和每个 rank 的通信角色。 [Megatron PP 消息形状与调度](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py#L1951-L1979)

## 第 4 章 核对单卡峰值

显存峰值按时间点计算。将静态训练状态与那一刻仍存活的激活、logits、通信缓冲及算子工作区相加，再看各 rank 的最大值。已算入训练状态的 DP 参数和梯度缓冲按一份计。

| 数值 | 本例的口径 |
| --- | --- |
| **65.16 GiB** | 静态训练状态 |
| **1.50 GiB** | 为反传保存的 48 份层输入 |
| **2.32 GiB** | 一份全序列 BF16 logits |
| **0.29 GiB** | 2,048 token 的一块 FP32 logits |
| **0.50 GiB** | 一次 EP dispatch 的平均收发缓冲 |
| **128 MiB** | 一次 TP 序列 All-Gather 的输出缓冲 |

例如，所有层输入还在，同时生成一份全序列 BF16 logits 时，**已计的三项是 65.16 + 1.50 + 2.32 = 68.98 GiB／卡**。EP、TP、DP、CP 和 PP 的新增缓冲按各自的存活时间计入。整步网络流量按调用次数和数据实际跨卡去向累计，显存预算查同一时间点的缓冲。

## 第 5 章 SGLang 推理显存与通信

继续用 Qwen3-30B-A3B，推理配置按 **SGLang `d2t4p1`** 估算。DP=2 表示 **两个独立的模型副本**，每个副本用 TP=4 切权重。PP=1，本例采用常规 TP 推理，CP=1，EP=1，关闭 DP Attention。总共 8 张推理卡。下面分别看单卡显存与副本内传输量。 [AReaL 推理配置语义](https://github.com/areal-project/AReaL/blob/f8c706b0bec865aafbbee67f59af507316f29965/docs/en/reference/alloc_mode.md#L81-L94) · [Qwen3 配置](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json)

### 5.1 模型权重

按各 PP stage 的可切分参数量近似均分估算，$P_{\mathrm{shard}}$ 是全模型可切分参数量，本卡份额用它除以 $\mathrm{TP}\times\mathrm{PP}$。$P^{(s)}_{\mathrm{replica}}$ 是当前 stage 在本卡复制的参数量。分层不均时，按该 stage 的实际参数量计算。本例 PP=1，路由专家的矩阵也随这一副本的 TP 切分，router 和 norm 等小量参数按复制估算。每个 DP 副本各自加载一份 TP 分片。

**式 14 · SGLang 单卡模型权重**

$$
\left(\underbrace{\frac{P_{\mathrm{shard}}}{\mathrm{TP}\times\mathrm{PP}}}_{\text{全模型可切分参数量}\,\div\,(\text{TP}\times\text{PP})}+\underbrace{P^{(s)}_{\mathrm{replica}}}_{\text{本卡复制参数量}}\right)\times\underbrace{2}_{\text{每个 BF16 权重的字节数}}
$$

代入 **（30.51933B 个可切分参数 ÷（TP 4 × PP 1）＋0.01279B 个本卡复制参数）× 2 字节（BF16）**，得到 **约 14.24 GiB／rank**。其中可切分参数量包含约 28.991B 路由专家参数与 1.52830B 其余可切分参数。

### 5.2 单条请求与并发请求的实际 KV 占用

一条请求的 KV token 数按**已经缓存的上下文长度**计入，包括已处理的 prompt 和已缓存的生成 token。$T_{\mathrm{cache}}$ 是当前副本实际驻留的缓存 token 数。若请求等长、均匀分到两个副本且各自使用独立 KV，先用**全局并发请求数 ÷ DP**得到每个副本的请求数，再乘**每条已缓存的 token 数**，得到 $T_{\mathrm{cache}}$。有共享前缀时，按实际驻留的缓存槽位估算。

**式 15 · 单卡实际占用的 BF16 KV**

$$
\underbrace{T_{\mathrm{cache}}}_{\text{本副本当前缓存的 token}}\times\underbrace{L_s}_{\text{本 PP stage 的层数}}\times\underbrace{2}_{\text{K、V 各一份}}\times\underbrace{n_{\mathrm{KV,local}}\,d_{\mathrm{head}}}_{\text{本卡 KV head 数}\times\text{每 head 宽度}}\times\underbrace{b_{\mathrm{KV}}}_{\text{每个 KV 元素的字节数}}
$$

模型有 4 个 KV head，TP=4 后每 rank 负责 **4 个 KV head ÷ TP 4 = 1 个**，每 head 128 维。在这组配置里，**每个缓存 token 占本卡 48 层（PP=1）× K/V 两份 × 1 个本地 KV head × 128 维 × 2 字节（BF16）= 24 KiB**。有 PP 时用当前 stage 的实际层数。层均分时可用 $L_s=L/\mathrm{PP}$。 [SGLang Qwen3-MoE 的 KV head 分片](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py#L465-L485)

| 负载均衡时的全局并发 | 每个副本的缓存量 | 单卡实际 KV 占用 |
| --- | --- | --- |
| 2 条 32K 请求 | 2 条 ÷ DP 2 = 1 条，32,768 个缓存 token | **0.75 GiB** |
| 16 条 32K 请求 | 16 条 ÷ DP 2 = 8 条，262,144 个缓存 token | **6.00 GiB** |

单条 32K 请求对应本卡的**模型权重 + 已占用 KV 约 14.99 GiB**。并行副本的请求数不均时，用最忙副本实际驻留的 token 数。

### 5.3 服务启动时的 KV 池预算

SGLang 会预分配实际请求可用的 KV 池。$f$ 对应 `--mem-fraction-static`，覆盖模型权重和 KV 池。下面是按 GPU 容量得到的预算。

**式 16 · 按静态显存比例估算 KV 池**

$$
\underbrace{M_{\mathrm{KV,pool}}}_{\text{预计的 KV 池预算}}\;\approx\;\underbrace{f}_{\text{SGLang 静态显存比例}}\times\underbrace{M_{\mathrm{GPU}}}_{\text{本卡显存容量}}\;-\;\underbrace{M_{\mathrm{weights}}}_{\text{本卡权重显存}}
$$

例如按 **80 GiB／卡 × 0.8 = 64 GiB** 估，减掉 **14.24 GiB 权重**，KV 池的名义预算约 **49.76 GiB／卡**。用 KV 池字节数除以 **24 KiB／缓存 token** 可以估槽位数。服务启动还会受其他分配、缓存页取整和运行参数影响，实际容量用日志里的 `max_total_num_tokens` 核对。**实际请求占用的 KV 来自池里**，估服务所占显存时按池的分配量计一次。 [SGLang `--mem-fraction-static` 说明](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hyperparameter_tuning.mdx#tune---mem-fraction-static-to-increase-kv-cache-pool-capacity)

### 5.4 Prefill 和 Decode 的临时显存

估临时工作区时看**当前调度轮次**。Prefill 用这轮实际送进前向的 token 数，包含同一批次里的分块。普通 Decode 每条已调度请求贡献一个新 token。下面算的是**一份完整 hidden state**。分片后的张量再按实际形状代入。

**式 17 · 单份 Prefill 或 Decode 隐藏张量**

$$
\underbrace{T_{\mathrm{step}}}_{\text{这轮实际调度的 token 数}}\times\underbrace{H}_{\text{每个 token 的隐藏维度}}\times\underbrace{2}_{\text{每个 BF16 元素的字节数}}
$$

例如本副本这轮实际处理 **4,096 个 token × 2,048 个隐藏维度 × 2 字节（BF16）= 16 MiB**。峰值还包括同一时刻存在的注意力、MoE 和采样工作区、CUDA Graph 缓冲等。Logits 用 **实际物化 logits 的位置数 × 该张量的词表宽度 × dtype 字节数** 估。连续批处理的生成路径可以只为需要采样的位置生成 logits，请求更多 token 的 log-prob 时要看相应处理分支。 [SGLang 分块 Prefill 与 CUDA Graph](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hyperparameter_tuning.mdx#avoid-out-of-memory-errors-by-tuning---chunked-prefill-size---mem-fraction-static-and---max-running-requests) · [SGLang logits 位置选择](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/logits_processor.py#L865-L923)

### 5.5 推理侧通信

当前估算中，两个外层 DP 副本各自处理请求。模型前向的 tensor collective 发生在各自的 TP 组内。CP=1、PP=1、EP=1，跨 CP、跨 stage 和专家 All-to-All 流量均为 0。当前若有一个对完整 hidden state 的 **TP Ring All-Reduce**，每个 rank 单向发出的字节数可按下式估算。这里 $b$ 是通信 tensor 的 dtype 字节数。

**式 18 · 一次 TP Ring All-Reduce 的单向传输量**

$$
\underbrace{T_{\mathrm{step}}\,H\,b}_{\text{要归约的本地 hidden state 字节数}}\times\underbrace{\frac{2(\mathrm{TP}-1)}{\mathrm{TP}}}_{\text{环形 Reduce-Scatter 与 All-Gather 各一轮}}
$$

若一轮处理 **4,096 token**，本地 BF16 hidden state 大小为 **16 MiB**。TP=4 的 Ring All-Reduce 系数为 **2 × (4−1) ÷ 4 = 1.5**，所以一次该通信约 **24 MiB／rank 单向发出**。接收方向同量。实际 SGLang 可能选用定制的规约或其他 collective，按实际算子、张量形状和调用次数估。这个传输量与 GPU 上另行占用的通信临时缓冲各算一笔。训推权重同步在权重更新时单独累计。 [NCCL All-Reduce 环形系数](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce)
