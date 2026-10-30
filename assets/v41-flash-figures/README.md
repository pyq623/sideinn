# V4.1-Flash 笔记配图

对应 `../../v41-flash.md`，资料基线 2026-10-26，论文固定为 arXiv 2609.19969v1（2026-09-17）。

## 论文原图

`official-*` 来自论文 HTML 的原始图像文件；未重绘、未裁切、未改字。Figure 1(b) 是原站单独提供的子图文件。下载地址、正文 figure anchor、SHA-256 与作者信息保存在 `official-sources.json`。技术讲解与中文图注在笔记正文。

论文使用 arXiv non-exclusive distribution license，图像权利归 DeepSeek-AI 等原作者。本目录不为原图另行授予许可；不要将站点代码或原创图的许可套用于论文原图。引用时保留原作者与论文链接。

## 原创静态教学图

运行 `python3 scripts/build-v41-flash-figures.py` 可从站点根目录重新生成下列 SVG；只依赖 Python 标准库，不修改论文图。

- `01-cost-map.svg`：新输入计算、历史容量、检索与恢复三种成本。
- `02-ced-paths.svg`：全长 global KV 与尾部 hidden states 的两条路径。
- `03-candidate-pool.svg`：人工 16 位置、4 位置一块、选 2 块、Top-2 的示例。
- `04-byte-ledger.svg`：根据论文配置计算 288 + 68 = 356；356 × 2.5 = 890 B/token。
- `05-bounded-replay.svg`：人工 W=4 的截断恢复；实际窗口 128。
- `06-async-rollout.svg`：共卡分时、样本跨检查点恢复的逻辑时序，宽度不表示测量时间。

上述图为解释机制而作，省略部分实现细节；不能作为模型完整计算图或性能测量。正文明确人工示例、配置推导与作者实验的区别。不含动画。
