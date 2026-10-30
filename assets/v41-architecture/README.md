# V4.1-Flash Figure 3 分步动画

入口：`notes/v41-architecture.html`。从 `v41-flash.md` 的 2.2 节进入。

- `steps.mjs`：22 步内容、原图坐标、路径、层号、独立缓存来源计数及论文定位。
- `player.mjs`：逐步演示、自动播放、暂停/继续、速度、深链接、聚焦与减少动态。使用 requestAnimationFrame；播放按实际经过的时间推进，隐藏页面立即暂停。
- `player.css`：桌面图文并排、手机上下排布与局部聚焦。

背景直接引用 `../v41-flash-figures/official-fig3-architecture.svg`，原始文件不改动。叠加层坐标使用原 SVG 的 910×469 viewBox。`steps.mjs` 的颜色只描述当前路径语义，光点代表数据依赖方向，不代表真实 token/算子时序或实测性能。

流程以长输入 Prefill 为主，再说明 Decode 与 DSpark。CED 全长缓存投影与尾部 hidden states 前向分开演示；尾部 128 的有限重放来自论文 §3.2.2，不是原图箭头自带的窗口含义。虚线重复组由实际层号展开，不表示循环使用同一层参数。独立 global KV 来源计数为 Encoder 3、Decoder 1；副本、SWA 和工作区不在此计数内。

验证重点：22 个步骤跳转、0–39 层连续展开、缓存计数、原图哈希、原文定位、自动播放/暂停/重播、速度、首尾边界、深链接刷新、减少动态、隐藏页面暂停、桌面/手机无整页横向溢出。演示与原图权利说明在页面末尾。源码不依赖构建或第三方网络脚本。

运行 `node scripts/check-v41-architecture.mjs` 验证步骤、真实层号展开、缓存来源、双路径类型与原图哈希。

## 第二步 mHC 内嵌图

`single-pass-mhc.svg` 沿用旧文件名，内容现为 Residual Connection 与基础 HC 的并排重绘；每组保持从下向上的原图方向。`mhc-single-pass-comparison.svg` 则在相同视觉样式下对照 mHC 与 Single-Pass mHC：当前输入混合由 A(l) 改用上一子层的 A(l−1)，当前预测的 A(l) 交给下一子层，B(l)/C(l) 仍用于当前子层。黑实线表示激活数据，蓝虚线表示输入混合系数；按 V4.1 §2.4.1 式 2/6 绘制。这两张图及文字解释放在独立页面 `notes/v41-single-pass-mhc.html`；第二步左侧简短链接和聚焦按钮左边的 Single-Pass mHC 入口均跳转到该页。页面提供返回第 2 步入口，窄屏在图解区域横向滚动。它们展示逻辑依赖，非 GPU kernel 边界。原先的外部图片弹层、热点及放大事件已移除。

### 数学排版

数学符号必须写标准 TeX，使用 MathJax 生成 SVG，禁止 Unicode 下标替代。图形源文件为 `single-pass-mhc.svg.in` 与 `mhc-single-pass-comparison.svg.in`；图下说明在独立讲解页 HTML 的 `span.mhc-math[data-tex]`。构建依赖 `mathjax-full@3.2.2`，可装在仓库外，执行 `NODE_PATH=<安装目录>/node_modules node scripts/build-v41-mhc-math.cjs`。输出 SVG 与页面内公式均为自包含矢量路径，无新增浏览器依赖。

### 当前图的内容（2026-10-29 更新）

用户要求复刻原参考图，当前 SVG 已替换为普通 Residual Connection 与 Hyper-Connections 两组整体/局部结构的横向重绘。保留原图英文模块名、中文解释及 TeX 符号。文件名沿用 single-pass-mhc.svg，但图本身明确是基础 HC 结构；Single-Pass 的区别在图下注释中说明。此前“四路进，一路算，四路出”的简化图已被替换。

布局纠正：单张结构保持原图的竖向（从下到上），普通结构组与 HC 结构组左右并排。不要把各模块流向旋转为水平，也不要将两组上下堆叠。

## 第四步 Engram 图解（2026-10-30）

第 4 步 `engram` 的左侧短链接与工具栏 Engram 按钮进入 `notes/v41-engram.html`，返回链接恢复 `#engram`。独立页复用 mHC 的 300px 说明栏与紧凑右侧图解布局。`engram-architecture.svg.in` 为可编辑源，构建脚本 `build-v41-mhc-math.cjs` 同时生成其 SVG 与页面中的 LaTeX 公式。

图依据 Engram Figure 1：左侧主干位置，右侧从下向上的多头查表与上下文门控。按 V4.1 §2.4.2 展示 2/3/4-gram、每种 8 个哈希头，去掉短卷积。内部展开单个 mHC 分支；说明中明确共享查表/Value、分别计算 Key/门控。桌面 1440px 图宽 912px；390px 手机无整页横向溢出，图内滚动。已验证进入与返回第 4 步、第 2 步 mHC 入口不受影响、22 步检查通过。

## 第五步 CSA2 图解（2026-10-30）

第 5 步 `encoder-full` 的工具栏 CSA2 按钮与左栏短链接进入 `notes/v41-csa2.html`。独立页沿用紧凑双栏布局；`csa2.mjs` 控制六步：隐藏状态、两位置压缩、索引准备、评分、选址取回、注意力读取。提供自动播放/暂停、前后步、重播、减少动态、深链接与隐藏页面暂停。`csa2.css` 仅作用于新页。

`csa2-flow.svg.in` / `.svg` 为本地教学图，由现有 MathJax 构建脚本生成并嵌入页面。查询位置9；完整对1–8形成四条缓存；Top-2/窗口3/评分是教学设定，真实配置为Top-512。两位置加权示例对应 DeepInfer Compressor 的逐维 softmax 加权，明确未展开后续归一化、RoPE、量化。检索键来自归一化后 RoPE 前的压缩表示。main Q 与 indexer Q 分开画，Gather 读取缓存内容，SWA 不参与 Top-K。

下方直接展示未修改的论文 Figure 4 与三模式分工表，并链接 DeepInfer 原动画。验证：六步高亮、选中地址2/4、自动完成与暂停、减少动态、返回第5步、其他步骤入口显隐、1440/390px 无整页溢出、原图正常加载、22步检查通过。数学源修改后执行既有 `build-v41-mhc-math.cjs` 重建。

CSA2 布局支持拖动两栏间分隔线调整说明宽度，默认240px，范围180–420px（窄容器还会限制上限）。方向键微调，Home/End到边界，双击或按钮恢复默认。偏好保存在 `sideinn:v41-csa2:sidebar-width`；桌面SVG随右栏宽度完整缩放，手机仍为单列与图内滚动。

架构主动画左侧说明栏在桌面与右图等高，长文字在栏内独立竖向滚动；切换步骤回到文字顶部。不要改成折叠面板。900px及以下为自然单列排版。

## 缓存示意与候选池入口（2026-10-30）

`cache-view.mjs` 按当前步骤显示三套 Encoder 和一套 Decoder 来源，每套分出 main KV 与 indexer K。第13步的“候选池图解”进入 `notes/v41-candidate-pool.html`；示例选Top-2位置与Top-2块，真实配置分别为Top-512 positions与Top-2048 blocks。页面顶部/底部返回第13步，原动画不再嵌入长示例。
