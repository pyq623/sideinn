# 支线客栈维护接手文档

核对日期：2026-10-19。本文描述当前实现；接手时先检查远端最新提交和实际 Pages 设置。

## 1. 先看这几条

- 正式网站：**https://sideinn.ink/**，由 GitHub Pages 免费托管。
- 正式站仓库：**https://github.com/pyq623/sideinn**。`main` 分支根目录就是发布目录，推送到 `main` 会触发上线。
- 基础、工程和面试资料源码：**https://github.com/pyq623/ai-study-notes**。修改这些资料要先改这个仓库，再构建、导入正式站。
- **两个仓库不会自动同步。** 只更新资料仓库，不会更新 `sideinn.ink`；只改正式站的 `study/*.html`，下次导入又会被覆盖。
- 普通改文字、改页面不需要修改域名、DNS、证书或购买服务器。
- 网站里的“编辑笔记”仅保存浏览器草稿，不能更新公开内容。

## 2. 源码在哪，应该改哪里

建议在一个工作目录里并排克隆两个仓库。以下命令仅用于首次建立工作目录；已有 checkout 时先检查 `git status` 和远端，不要覆盖现有工作。

```sh
git clone https://github.com/pyq623/sideinn.git sideinn
git clone https://github.com/pyq623/ai-study-notes.git ai-study-notes
```

两个仓库都公开可读；推送需要有对应权限的 GitHub 账号。不要在源码、文档或命令输出里保存访问令牌。

### 正式站仓库 `sideinn`

| 要修改的内容 | 文件或目录 |
| --- | --- |
| Hot100 目录、逐题正文、图片与复习标记 | `hot100/`；从用户 DOCX 用 `scripts/import-hot100.py` 导入，生成内容不手改；来源摘要见 `hot100/import-source.json` |
| 首页文案、结构、资源版本参数 | `index.html` |
| 首页分类、搜索、卡片渲染 | `home.js` |
| 首页布局和卡片封面 | `home.css` |
| 首页目录数据：标题、分类、标签、简介、入口、更新时间 | `assets/notebook.json` |
| 全站阅读器共用颜色变量 | `theme.css` |
| 面试、笔试及工程笔记的本站阅读器外观与目录交互 | `study-reader.css`、`study-reader.js`；由导入器注入，保留上游练习逻辑 |
| 研究笔记阅读器布局 | `styles.css` |
| 研究笔记渲染、搜索、目录、下载、草稿 | `app.js`、`paragraph-core.js` |
| RSI 正文 | `notes.md` |
| 推理基础设施正文 | `inference-infra.md` |
| 研究笔记的页面壳、元信息与数据源配置 | `notes/rsi.html`、`notes/inference-infra.html`、`notes/occamy.html` |
| SAO / CompactionRL 正文、公式与配图 | `sao.md` / `compactionrl.md`、`assets/*-formulas.json`、`assets/sao-figures/` / `assets/compactionrl-figures/`；阅读页为 `notes/sao.html` / `notes/compactionrl.html` |
| Occamy 专题正文、公式与配图 | `occamy.md`、`assets/occamy-formulas.json`、`assets/occamy-figures/` |
| RSI / 推理笔记的编号公式与窄屏版本 | `assets/formulas.json` / `assets/infra-formulas.json` |
| 图示 | `assets/diagrams/`、`assets/infra-diagrams/` |
| 已导入的学习、工程、面试页面 | `study/`，属于生成结果 |
| 将资料导入并适配到本站的逻辑 | `scripts/import-study.py` |
| 当前导入的上游提交 | `study/import-source.json` |
| 域名与静态托管标记 | `CNAME`、`.nojekyll` |

这是普通 HTML/CSS/JavaScript 静态网站，正式站根目录没有 `package.json`，不需要 `npm install` 或前端构建。Marked 和 MathJax 随站点放在 `assets/vendor/`。

`study/` 保留上游 `study/assets/course.css`、预渲染 KaTeX 和练习脚本；导入器在其后加载本站 `theme.css`、`study-reader.css`、`study-reader.js`，统一纸面布局、侧栏、配色和手机目录。研究笔记布局仍由 `styles.css` 控制。修改本站外观优先调整阅读器适配层，不手工修改生成页面。

### 资料源码仓库 `ai-study-notes`

| 要修改的内容 | 文件或目录 |
| --- | --- |
| 机器学习、线代、概率、神经网络、Infra、数据结构、计算机系统、强化学习 | `content/unit-1.md` 至 `content/unit-8.md` |
| Agent 与工程实践 | `content/engineering.md` |
| 基础 / 强化学习 / 工程练习 | `content/foundation-questions.json`、`content/rl-questions.json`、`content/engineering-questions.json` |
| 面试讲义、60 道题、20 道口述追问与来源 | `content/interview-course.json` |
| 面经整理 | `content/interview-research.html` |
| 工程资料来源、关于与许可说明 | `content/engineering-sources.json`、`content/about.md` |
| 页面结构、目录、跨章链接、存储键配置 | `scripts/build.py` |
| 面试页面和打印总册生成 | `scripts/interviews.py` |
| 共用题卡及答案 HTML | `scripts/question_rendering.py` |
| 学习与面试页面样式 | `assets/course.css` |
| 基础练习 / 面试练习交互 | `assets/quiz.js` / `assets/interview-quiz.js` |
| 答题记录兼容、搜索、旧锚点迁移 | `assets/records.js`、`assets/site.js`、`assets/legacy-links.js` |
| 验证 | `tests/` |
| 构建产物 | `docs/`，被 Git 忽略，不是手工维护的源码 |
| 旧网址的自动构建发布 | `.github/workflows/pages.yml` |

## 3. 常用维护流程

### A. 修改首页或两篇研究笔记

1. 在 `sideinn` 中检查工作区、分支和最新远端，保留已有未提交修改。
2. 按上表编辑对应文件。正文、阅读页标题/日期、首页简介和统计需要保持一致。
3. 修改公式时，同步相应 JSON 中的 `number`、`wide`、`narrow`。渲染器通过去空白后的公式文本匹配目录；正文 LaTeX 与 `wide` 不一致会失去编号和窄屏替换。
4. 改动 CSS/JS 后，更新引用它们的 HTML 中的 `?v=...` 参数，减少浏览器继续使用旧资源的情况。
5. 本地预览、检查受影响功能，然后按照第 5 节发布。

新增研究笔记时，复制现有 `notes/*.html` 页面壳，配置唯一的 `data-note-id`、`data-note-source`、`data-formula-source` 和下载文件名，并在 `assets/notebook.json` 增加入口。不要复用其他笔记的 ID，以免浏览器草稿冲突。

首页每条目录数据需要 `id`、`href`、`title`、`english`、`category`、`tags`、`summary`、`updated`。统计可以使用 `chapters/formulas/references`，或使用集合型条目的 `details`；封面和按钮可配置 `coverLabel`、`coverFooter`、`readLabel`。检查手机宽度下长封面文字是否被截断。

### B. 修改面试、笔试或工程资料

先在 `ai-study-notes` 修改源码，再把生成的静态页面导入 `sideinn`。

源码仓库要求 Python 3.11+、Node.js 20+。首次准备环境，在 `ai-study-notes` 根目录运行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
npm ci --ignore-scripts
```

每次修改后的流程：

1. 编辑源码，运行 `npm run build` 和 `npm test`，处理实际失败。
2. 检查并提交上游源码改动；将构建来源落实为可追溯的提交。若通过 PR 合并，切换到合并后的提交并重新构建。
3. 在来源 checkout 干净、构建内容与 HEAD 一致时执行导入。脚本只记录 `git rev-parse HEAD`，不会识别未提交源码或检查构建是否过期。
4. 在 `sideinn` 根目录执行以下命令；路径按实际目录调整：

```sh
python3 scripts/import-study.py ../ai-study-notes
```

5. 检查 `study/import-source.json` 的提交号，检查 `git diff`。如题数、章节、简介有变化，同步 `assets/notebook.json` 和相关说明。
6. 在正式站的本地预览中验证，再提交并发布正式站仓库。

上游 `main` 的推送会发布原来的 `https://pyq623.github.io/ai-study-notes/`；正式站的 `main` 推送才会发布 `https://sideinn.ink/`。修改两个站点时，应记录两个仓库对应的提交。

导入器保留题目、题号、图示、公式和练习逻辑，调整站名、阅读器布局、章节侧栏、返回首页导航、页脚说明及旧 404 基础路径。本站专用适配应写进导入器，避免手工改 `study/` 后丢失。

**导入器的现有限制：**

- 使用目录覆盖复制，不会清理上游已删除的旧文件。删除/改名章节时，要核对并显式删除对应的废弃生成文件和入口。
- 不会自动构建或运行测试，也不会提交、推送或更新首页统计。
- 如果上游页面壳改变，可能报 `Source navigation changed`。此时核对模板并更新导入器，不要直接忽略错误。
- 不是原子操作，失败时可能已经写入部分文件。修正后完整重跑并检查差异，不要发布半成品。

## 4. 验证与本地预览

在 `sideinn` 根目录启动：

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

打开 `http://127.0.0.1:8000/`。使用 HTTP 预览；双击 HTML 的 `file://` 方式不能代表真实运行情况。结束后停止自己启动的服务。

根据改动范围验证，不必每次重跑所有检查：

- 首页：分类筛选、搜索、空结果恢复、所有卡片链接。
- 研究笔记：正文加载、目录锚点、公式和图片、搜索、手机目录、Markdown 下载；修改编辑器时验证草稿刷新恢复。
- 学习和面试：选择答案与把握程度、正确/错误判分、解析、重做、错题筛选、刷新后记录、JSON 导出。
- 面试专属：口述笔记保存、只看习题、打印总册、跨章与来源链接。
- 页面变动至少检查 390px 手机宽度和桌面宽度；卡片封面另看 520px 左右的中间布局。长公式和表格应在容器内滚动。
- 检查浏览器脚本错误、资源 404、站内链接和锚点。不要只凭“首页能打开”判断全部完成。
- 使用独立测试浏览器或新的测试上下文，不覆盖用户自己的答题记录。支持练习预览记录的页面可用 `?preview=1` 隔离练习数据。

基础静态检查，在 `sideinn` 中运行：

```sh
node --check home.js
node --check app.js
python3 -m json.tool assets/notebook.json > /dev/null
git diff --check
```

涉及资料源码时，在 `ai-study-notes` 激活虚拟环境后运行：

```sh
npm run build
npm test
```

`sideinn` 当前没有提交通用浏览器自动化测试套件。迁移时使用的临时本机 QA 脚本不属于仓库依赖，接手 agent 应使用自己环境中可用的浏览器工具执行上述检查。

2026-10-19 的验证基线：首页 4 个入口；原研究笔记 18 + 35 个编号公式；学习练习 208 道，面试练习 60 道，口述追问 20 道。资料目录有 21 个正常 HTML 页面和 1 个 404 页面。该次上游验证通过 33 个 Python 测试和 5 个 Node 测试，且完成手机与桌面交互检查。内容增长后应更新基线，不应为了凑旧数量删题。

## 5. 发布、确认与回退

接手先运行 `git status --short`、`git remote -v`，确认当前仓库和已有改动。需要新分支时使用 `codex/` 前缀；是否直接推送 `main` 或走 PR，按用户当次授权和仓库实际保护规则执行。

提交前查看差异，明确选择要提交的文件。上级工作目录可能有其他私人项目，不能把整个工作目录一起提交。

当前正式站设置：

| 配置 | 值 |
| --- | --- |
| Pages 发布方式 | Deploy from a branch（API `build_type: legacy`） |
| 分支 / 目录 | `main` / `/` |
| 自定义域名 | `sideinn.ink` |
| HTTPS 强制 | 已开启 |
| 静态处理标记 | 根目录 `.nojekyll` |

推送或合并到 `main` 后，在 [Actions](https://github.com/pyq623/sideinn/actions) 等待 **该提交** 的 `pages-build-deployment` 成功。可以使用：

```sh
gh run list --repo pyq623/sideinn --limit 5 --json databaseId,headSha,status,conclusion,url
gh api repos/pyq623/sideinn/pages
```

根据列表确认正确的 run ID，再用 `gh run watch RUN_ID --repo pyq623/sideinn --exit-status` 等待。不要将上一条成功任务当成本次成功。

发布后访问正式域名的首页和实际修改页面，核对新内容、关键资源及交互。若仍显示旧版，先确认提交和部署，再检查浏览器/CDN缓存；不要因此修改 DNS。报告已完成的验证，以及无法从当前环境判断的事项。

回退时优先对明确的错误提交执行 `git revert`，生成新的回退提交，再正常发布；不要使用强推覆盖他人的历史。若改动跨两个仓库，分别判断需要回退的源码和生成内容，保留可追溯关系。

## 6. 域名、费用与访问边界

域名由用户在阿里云管理，托管在 GitHub Pages。当前阿里云根域名 `@` 的 A 记录为：

```text
185.199.108.153
185.199.109.153
185.199.110.153
185.199.111.153
```

TTL 为 10 分钟；`CNAME` 文件只有 `sideinn.ink`。HTTPS 由 GitHub Pages 管理，无需自己上传证书。正常内容维护无需登录阿里云控制台。

旧平台留下的 `_cf-custom-hostname`、`_openai-site-verification` TXT 记录在本次迁移中保留了。不要将历史 Cloudflare IP 当成当前发布地址恢复，也不要顺手清理不属于本次任务的 DNS。

当前无需新增托管费用；域名续费仍由用户负责。主站仓库公开，禁止把私有笔记、聊天、浏览器记录、数据库备份、凭据或其他私人工作区材料放进发布目录。

用户曾确认原 GitHub Pages 地址在其大陆手机网络上可直连；这不代表所有地区、运营商和时间都稳定。代理环境或境外环境的成功不能证明大陆手机直连成功。访问问题应先区分具体网络错误、DNS、HTTPS、404 和页面脚本错误，再定位。

## 7. 数据、兼容和历史系统

- 网站没有账号同步和数据库后端。研究笔记草稿、答题、口述内容保存在 `localStorage`，仅属于当前浏览器与域名。
- 更换域名、设备或清理浏览器后，记录不会自动转移。练习支持导出 JSON，当前没有导入功能；不要承诺导出文件能一键恢复。
- 不要随意更改题号、章节锚点、`data-note-id` 或存储键。源码 `scripts/build.py` 中工程专题内部 ID 为 8，新展示单元 8 的内部 ID 为 9；工程专题保留历史 `unit-6-v2` 存储键。这些并非可随手统一的命名错误。
- 面试题保留 `AR1-01` 至 `AR5-12` 的题号，以及 `accio-agentic-rl-v1-u1` 至 `u5` 的存储键。
- 原 OpenAI Sites 项目和旧评论数据库属于历史系统，未包含在当前公开仓库里。它们不是 `sideinn.ink` 的当前发布源。当前站点显示“在线评论暂未开放”；不要通过重新发布旧平台来更新本网站。
- `study/` 的许可与来源文件、KaTeX 许可，以及根目录第三方资源许可需要保留。不同区域的许可不应相互扩大适用范围。

## 8. 可直接交给下一位 agent 的任务说明

> 请维护 https://sideinn.ink/。先阅读主仓库 https://github.com/pyq623/sideinn 中的 AGENTS.md 和 HANDOFF.md，并检查当前分支、远端与未提交改动。首页及研究笔记直接在 sideinn 修改；面试、基础和工程资料请在 https://github.com/pyq623/ai-study-notes 修改源码，构建与测试通过后，用主仓库 scripts/import-study.py 导入。保留题号、存储兼容和来源许可，不要把生成页面当成长期编辑源。根据本次需求完成手机/桌面验证，并在授权范围内发布，核对该提交的部署和正式网址。无需为普通页面修改更改 DNS、证书或购买服务。最后报告修改位置、两个仓库涉及的提交、验证结果及尚存限制。

本次接手基准：合并资料的正式站提交 `5a8ff524096fb9f63cfb4787ffe02ef2ac641fa1`；对应资料源码提交 `01a64eaaf5d6839c46edc5d02ffc4b8f75dc4efb`。以后以远端最新状态和 `study/import-source.json` 为准。

### Hot100 导入与核对

2026-10-21 本地新增独立「算法与 Hot100」分类。原手册正文为 95 道编号题 + 1 道补充题，共 96 道题，另有 4 篇技巧总结；不是额外补写 100 道题。导入器跳过旧纸面目录，使用唯一 LeetCode 题号建立逐题页面，保留 68 张图片（72 次引用）、代码换行缩进和表格。更新时运行 `python scripts/import-hot100.py /path/to/Hot100算法题复习手册.docx`（依赖 lxml、Pillow），核对内容数量与内链，再检查首页分类、搜索、手机排版和复习标记持久化。复习记录使用独立键 `sideinn:hot100:review:v1`。首页以独立分类展示；正式入口为 `/hot100/`。


### DeepInfer 笔记与连续演示（2026-10-23）

- 首页入口在 `assets/notebook.json`；正文 `deepinfer.md`，阅读页 `notes/deepinfer.html`，公式 `assets/deepinfer-formulas.json`，样式 `assets/deepinfer.css`。沿用 Occamy 阅读器，独立草稿键 `study-notes-draft-v1-deepinfer`。共 13 章、42 小节、10 组公式、8 张官图和 5 张教学图。
- 图片与出处在 `assets/deepinfer-figures/`，官图来源、图号、哈希和许可见 `official-sources.json`。CED 图的数学符号为 MathJax SVG 路径；源文件 `ced-math.json`，生成器 `scripts/build-deepinfer-ced.py`。
- 第 3.1 / 3.4 / 4.1 节链接到 `notes/deepinfer-walkthrough.html`。23 页、4 节连续动画支持手动上一页/下一页、自动放映/暂停、速度、进度、章节跳转和放大。数值附录保留在 `notes/deepinfer-walkthrough-details.html`。
- 演示源码在 `assets/deepinfer-walkthrough/`：`model.mjs` 为唯一数值来源，`movie.mjs` / `movie.css` 为动画，`view.mjs` / `walkthrough.css` 为数值附录，维护说明见该目录 README。它使用人工 6 层二维模型，不是训练过的 DeepSeek 权重；第 9 token 由外部指定，短后缀重放明确标为近似。
- 机制依据 DeepSeek-V4.1-Flash 论文 `2609.19969v1`、DSpark `2607.05147v1`、官方部署文章及 DeepInfer 固定提交 `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` 的源码与 Q&A。未找到以该框架为题的独立原论文，勿与同名项目混淆。
- 维护后运行 `node scripts/check-deepinfer-walkthrough.mjs`，检查全长重放、截断误差传播、前缀缓存保留、增量 decode、因果性、索引复用和权重归一化；同时检查桌面/手机页面、全部 23 页、公式、手动/自动/暂停、正文图与首页入口。
- 本次按用户授权发布 DeepInfer 专题及首页入口；其他本地 Hot100、注意力代码和首页草稿不属于此发布范围。

2026-10-23 用户确认发布动画阅读改进：控制栏移到讲解上方，补齐 CED 两路和 Top-k 选择说明，主放映收敛为 4 节 23 页。第四节同时展示前缀 1–8 全局缓存全部复用、位置 1–5 不重算和尾部 6–8 重放恢复局部 KV；旧 #replay 跳转到该节。详细维护说明见 assets/deepinfer-walkthrough/README.md。

### DSpark 调度与验证动画（2026-10-23）

- 页面 `notes/deepinfer-dspark.html` 共 4 节、23 页：生成草稿、计算调度、目标验证、交付与下一轮。支持手动翻页、自动放映/暂停、速度、进度和章节跳转；草稿流程图支持放大。
- 源码在 `assets/deepinfer-dspark/`：`model.mjs` 为教学数值来源，`player.mjs` / `player.css` 为放映内容与样式，`draft-diagram.mjs` 绘制并行骨干与轻量顺序头流程。概率、随机数、耗时均为人工教学设定，没有运行真实模型或 GPU 性能测试。
- 每枚草稿分别显示调度估计 `c` 与前缀存活率 `a`，验证时显示目标/草稿概率 `p` / `q` 和接受概率。`R` 是请求数，`B` 是本轮目标前向的 query token 总数，不能与名为 B 的请求混淆；未调度草稿不填造验证概率。
- 正文 `deepinfer.md` 第 3、5 章分别用醒目卡片链接 CED 与 DSpark 动画，第 6 章链接 DSpark 的 `#schedule`；卡片样式在 `assets/deepinfer.css`。本次范围是 DSpark 动画及这些正文入口。
- 修改后运行 `node scripts/check-deepinfer-dspark.mjs`，核对预算最优值、连续前缀分配、接受/拒绝、Markov 修正、残差归一化和目标分布一致性；涉及 CED 时另跑 `node scripts/check-deepinfer-walkthrough.mjs`。浏览器检查桌面/手机全部 23 页、公式、流程图放大、手动/自动/暂停、章节跳转与正文入口。
- 当前发布分支已完成数值检查及桌面/手机全页验证；正式上线仍需合并后按第 5 节核对对应部署与正式页面。
