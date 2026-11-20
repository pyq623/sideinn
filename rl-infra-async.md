# 06 · 异步 RL、资源调度与权重同步

> **状态：待写入。** 本页为学习大纲，讲解、推导、源码走读和实验结论尚未写入。规划日期：2026-10-29。

[返回路线总览](./rl-infra-roadmap.html) · 建议进度：第 8–9 周

**学习目标：** 能够设计有版本语义、可恢复、可控制陈旧度的异步 RL 管线，并解释吞吐收益的算法代价。

**学习方法：** 论文机制 → 数学/成本模型 → 源码实现 → 小实验 → 大规模验证方案。

## 第 1 章 系统拆分与比较坐标（待写入）

- 沿 Occamy 基础，区分资源 placement、工作 scheduling 与数据 persistence 三个维度。
- 对比 slime、verl、AReaL 的训练/采样拓扑、版本边界和样本交接。
- 画出 prompt、trajectory、reward、logprob 与权重的生命周期。

## 第 2 章 异步队列与采样偏差（待写入）

- 推导生产速率、消费速率、队列长度、样本年龄和背压之间的关系。
- 比较时间陈旧度与版本陈旧度；分析长轨迹、失败重试和先完成先训练的长度偏差。
- 在固定 GPU 预算下比较 colocated、disaggregated 与动态资源分配。

## 第 3 章 权重发布、传输与加载（待写入）

- 设计 version ID、publish/load/ACK、可见性屏障和 sampler 切换协议。
- 拆解全量与 delta 同步的字节数、编码/解码、基线管理和额外副本成本。
- 追踪 CUDA graph、KV/state 与权重版本切换的一致性要求。

## 第 4 章 持久化、故障与可观测性（待写入）

- 研究 trajectory lineage、去重、重复消费、部分写入与重放语义。
- 设计 ACK 丢失、worker 重启、版本跳跃、delta 基线不匹配的恢复状态机。
- 为吞吐、GPU 空闲、队列年龄、版本差、丢弃率与有效更新建立联合指标。

## 第 5 章 无 GPU 小实验（待实现）

- 实现 CPU 离散事件模拟器：固定资源总量，改变生成长度分布、训练耗时、同步开销与 staleness 上限。
- 实现最小参数发布协议，并注入重复消息、ACK 丢失、worker 重启和 delta 基线错误。
- 交付：同步/异步对照图、事件日志与协议状态机；模拟结果不宣称真实模型收敛或 GPU 加速。

## 第 6 章 论文、报告与源码入口

以下为规划时的阅读入口。正式写入时需记录论文版本、访问日期和仓库 commit；`main` / `latest` 链接会变化，不代表已完成该版本的复现。

- [slime：RL Systems](https://thudm.github.io/slime/advanced/rl-systems.html)
- [slime：Rollout Scheduling](https://thudm.github.io/slime/advanced/rollout-scheduling.html)
- [slime：Delta Weight Sync](https://thudm.github.io/slime/advanced/delta-weight-sync.html)
- [slime：fully_async_rollout.py](https://github.com/THUDM/slime/blob/main/slime/rollout/fully_async_rollout.py)
- [verl：异步训练器](https://verl.readthedocs.io/en/latest/advance/v1_async_trainer.html)
- [AReaL 源码](https://github.com/areal-project/AReaL)
- [Accio-Lab Dressage](https://github.com/Accio-Lab/Dressage)

## 第 7 章 交付物与验收（待完成）

- 待写入：核心概念与公式推导。
- 待写入：固定版本的源码调用链、shape / ownership / 时序表。
- 待完成：可重复的小实验、反例和结果解释。
- 待写入：GPU 实测方案、对照组与适用边界。
- 验收问题：用明确的时间线解释一个样本由哪个策略生成、被哪个训练版本消费。
- 验收问题：证明恢复流程不会静默加载错误 delta 基线，并报告异步吞吐与偏差的取舍。

### 7.1 后续记录模板

- 本次阅读版本与来源：待填写。
- 新理解与仍不确定的问题：待填写。
- 实验条件、原始数据与结论：待填写。
- 下一步与需要 GPU 验证的项目：待填写。

[← 上一篇：05 · MoE 通信、GPU 执行与弹性并行](./rl-infra-moe.html)

[下一篇：07 · 大厂训练报告与系统设计综合 →](./rl-infra-reports.html)
