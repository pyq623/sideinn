# 05 · MoE 通信、GPU 执行与弹性并行

> **状态：待写入。** 本页为学习大纲，讲解、推导、源码走读和实验结论尚未写入。规划日期：2026-10-29。

[返回路线总览](./rl-infra-roadmap.html) · 建议进度：第 7 周

**学习目标：** 能够根据路由分布和硬件拓扑判断 MoE 瓶颈，并设计可测量的通信与执行优化。

**学习方法：** 论文机制 → 数学/成本模型 → 源码实现 → 小实验 → 大规模验证方案。

## 第 1 章 专家路由与通信账本（待写入）

- 逐步展开 dispatch、permute、expert GEMM、unpermute 和 combine。
- 计算 token 数、top-k、hidden size、精度对通信字节和专家 batch 的影响。
- 比较节点内/节点间 topology、wide EP 与不同专家放置方式。

## 第 2 章 负载不均衡与尾延迟（待写入）

- 区分平均负载、最大专家负载、rank 等待和端到端尾延迟。
- 研究专家副本、路由偏斜及动态负载均衡的收益与迁移成本。
- 比较训练、prefill 和 decode 的吞吐/延迟目标。

## 第 3 章 GPU 执行与重叠边界（待写入）

- 补齐 roofline、HBM 带宽、occupancy、Tensor Core、stream 与 event 依赖。
- 拆解通信计算 overlap 与 DBO，明确额外 buffer、拆分粒度和调度开销。
- 研究 CUDA graph 捕获限制、动态 shape 与 breakable graph 的边界。

## 第 4 章 弹性 EP 的一致性协议（待写入）

- 拆分 prepare、状态迁移、commit、请求排空和失败回滚。
- 区分暂停窗口与完整扩缩容耗时；建立显存和带宽峰值账本。
- 设计路由表、专家权重和在途请求跨版本一致性的检查。

## 第 5 章 无 GPU 小实验（待实现）

- 用 CPU 生成均匀、热点及随时间变化的路由 trace，比较专家放置与副本策略。
- 结合拓扑和带宽参数，模拟 dispatch/combine 的最大 rank 开销与尾部。
- 交付：成本模型与消融表；另列真实 GPU 上所需的 profiler 指标、时间线和测量方法。

## 第 6 章 论文、报告与源码入口

以下为规划时的阅读入口。正式写入时需记录论文版本、访问日期和仓库 commit；`main` / `latest` 链接会变化，不代表已完成该版本的复现。

- [Megatron：token dispatcher](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py)
- [SGLang：Breakable CUDA Graph](https://www.sglang.io/blog/breakable-cuda-graph)
- [vLLM：Elastic EP near-zero downtime](https://vllm.ai/blog/2026-10-19-elastic-ep-near-zero-downtime)
- [DeepInfer：部署问答](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md)

## 第 7 章 交付物与验收（待完成）

- 待写入：核心概念与公式推导。
- 待写入：固定版本的源码调用链、shape / ownership / 时序表。
- 待完成：可重复的小实验、反例和结果解释。
- 待写入：GPU 实测方案、对照组与适用边界。
- 验收问题：给出计算受限、带宽受限与负载偏斜三种场景的可区分指标。
- 验收问题：解释 near-zero downtime 指标的计时边界，不将其等同于扩容总成本为零。

### 7.1 后续记录模板

- 本次阅读版本与来源：待填写。
- 新理解与仍不确定的问题：待填写。
- 实验条件、原始数据与结论：待填写。
- 下一步与需要 GPU 验证的项目：待填写。

[← 上一篇：04 · vLLM / SGLang 调度与混合状态缓存](./rl-infra-serving.html)

[下一篇：06 · 异步 RL、资源调度与权重同步 →](./rl-infra-async.html)
