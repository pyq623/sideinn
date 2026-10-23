# CED / bounded replay 教学演示

主入口：`../../notes/deepinfer-walkthrough.html`，原笔记第 3.1 / 3.4 / 4.1 节提供入口。当前共 23 页、4 节：输入与 encoder → 两路进入 decoder → 继续生成 → 前缀复用与重放。数值附录为 `../../notes/deepinfer-walkthrough-details.html`。

## 文件与数据

- `model.mjs`：唯一数值来源。确定性人工模型，6 层、二维、单头、窗口 3、Top-2、K/V 绑定、残差系数 0.3。encoder 第 2 层从第一层输出投影 A，decoder 第 4 层从第三层输出投影 C；主 Q 与索引查询分离。
- `movie.mjs` / `movie.css`：23 页主动画及其样式。播放控制在章节导航和讲解上方，支持上一页/下一页、自动放映/暂停、速度、进度、章节跳转、键盘方向键、放大、减少动态效果和页面隐藏暂停。
- `view.mjs` / `walkthrough.css`：11 步数值附录，支持重放长度和位置选择、矩阵与逐层表格。

## 讲解与图示

CED 分叉先说明两路用途。第 9 页显示索引评分 → Top-2 位置选择 → 主注意力读取 KV，分数和位置来自模型 traces；第 11 / 13 页分别展示 Reindex / Reuse。局部窗口不参加该 Top-k；索引分数不是注意力权重。本例用 C 简化 indexer K。

主放映不包含计算差异的对照页；decoder prefill 完成后直接进入生成。第四节固定显示完整旧前缀 1–8 和全部已复用的全局缓存，1–5 各层标「不重算」，6–8 描边并以黄色重放。它们的全局条目同样复用，重放用于准备缺失的局部 SWA 状态。旧 `#replay` 链接转到第四节。

## 依据与边界

机制依据论文 v1 §2.2 / §2.3 / §3.2.2 / §4.2.2，以及 DeepInfer 固定提交 `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` 的 attn.py 和官方部署 Q&A。

权重、token 切分、6 层排列、窗口、候选池和简化算式都是教学设定，不是 DeepSeek 权重、激活或性能复现。未实现训练、LM head、采样、量化、MoE、RoPE、mHC；第 9 个 token 由外部指定。模型公式使用本地 MathJax SVG 排版，避免重复显示辅助 MathML。

## 验证

运行 `node scripts/check-deepinfer-walkthrough.mjs` 检查数值不变量。页面检查覆盖桌面 1440 px / 手机 390 px、全部 23 页、四个章节入口、手动翻页/自动播放/暂停/进度末页、旧锚点、Top-2 分数和缓存复用标签；检查数学错误、资源失败和整页横向溢出。
