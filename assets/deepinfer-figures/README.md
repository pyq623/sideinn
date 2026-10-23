# DeepInfer 原创教学图

五张 SVG 为本站根据原始资料独立绘制的概念示意，不是官方架构原图或性能测量。

- 01：DeepInfer README 与部署 Q&A 的逻辑组件关系。
- 02、03：DeepSeek-V4.1-Flash 论文 v1 的 CED、CSA2 与 bounded SWA replay。
- 04：DSpark 论文 v1 的累积接受率与固定预算分配；数值为手算示例。
- 05：CPU/设备重叠教学例；2 ms 与 8 ms 不来自基准。

完整来源与固定提交见 `../../deepinfer.md`，整理日期 2026-10-22。

## 官方原图（2026-10-23 补充）

`official-*` 为官方原始文件，未重绘、裁剪或改写图中文字。它们与本站原创 `01-*` 至 `05-*` 分开标注；页面每图附出处和中文讲解。精确下载地址、论文图号、访问日期、原始尺寸、SHA-256 与许可记录见 `official-sources.json`。

| 文件 | 原始来源 | 页面位置 |
| --- | --- | --- |
| `official-a-system.png` | DeepInfer `doc/images/architecture_overview.png`，部署 Q&A | 1.3，服务架构及请求/KV 两类箭头 |
| `official-b-prefill.png` | DeepInfer `doc/images/prefill_pareto.png`，README | 2.3，新计算输入吞吐与单步延迟 |
| `official-c-decode.png` | DeepInfer `doc/images/decode_pareto.png`，README | 2.3，每卡吞吐与单用户速度；说明接受率换算 |
| `official-d-v41-architecture.svg` | V4.1-Flash `2609.19969v1`，Figure 3 | 3.1，encoder/decoder 与模型模块 |
| `official-e-csa2.svg` | V4.1-Flash `2609.19969v1`，Figure 4 | 3.2，Full/Reindex/Reuse |
| `official-f-indexer.svg` | V4.1-Flash `2609.19969v1`，Figure 5 | 3.3，分层索引与候选池 |
| `official-g-dspark.svg` | DSpark `2607.05147v1`，Figure 1 | 5.2，草稿、调度与目标验证 |
| `official-h-wide-ep.png` | DeepInfer `doc/images/wide_ep_decode_pareto.png`，部署 Q&A | 7.1，EP8/EP64 对比与变量边界 |

仓库图固定提交 `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`，随附上游 Apache-2.0 文本 `LICENSE-DeepInfer.txt`。DSpark 图署名 Xin Cheng 等（Peking University / DeepSeek-AI），论文标注 CC BY 4.0。V4.1-Flash 图署名 DeepSeek-AI；arXiv 页面标注 non-exclusive distribution license，不应把它写成 CC BY，也不将本站自绘图或代码的许可扩大到论文图。此处将少量相关原图用于逐图技术讲解，图的权利归原作者。

## CED 图重绘（2026-10-23）

页面图 2 使用 `02-ced-latex.svg`，替代旧 `02-ced.svg`。绿色表示全局缓存路径，橙色表示隐藏状态及其查询/局部键值路径；两路显式接入注意力模块，并标明有限重放的近似性。公式原始 TeX 与 MathJax 输出保存在 `ced-math.json`，数学字形为独立矢量路径，不依赖查看设备安装数学字体。布局源为 `../../scripts/build-deepinfer-ced.py`，执行 `python3 scripts/build-deepinfer-ced.py` 可重建 SVG。若修改公式，应同步从站点 MathJax 的 `tex2svgPromise` 更新对应 TeX 和 SVG。
