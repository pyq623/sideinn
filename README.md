# 支线客栈

收留好奇心，记录未完成的探索。关于强化学习、智能体与推理系统的中文笔记。

正式网址：https://sideinn.ink/

GitHub Pages 入口：https://pyq623.github.io/sideinn/ （自动跳转到正式网址）

维护接手请先阅读 **[HANDOFF.md](HANDOFF.md)**：包含两套源码的位置、编辑与导入流程、预览验证、发布回退和域名配置。Agent 的简短入口为 [AGENTS.md](AGENTS.md)。

## 内容与功能

- RSI 递归自我改进：13 章、18 个公式。
- vLLM、SGLang 与推理基础设施：18 章、35 个公式。
- DeepInfer 笔记配套 [CED 连续演示](notes/deepinfer-walkthrough.html)和 [DSpark 调度与验证动画](notes/deepinfer-dspark.html)，各 4 节、23 页，支持手动翻页与自动放映。DSpark 沿用两条请求讲解草稿生成、预算分配、目标验证和交付；概率与耗时为教学设定。
- AI 基础与工程笔记：8 个主题单元和工程专题，208 道交互练习。
- Agentic RL 面试笔记：5 个单元、60 道练习、20 道口述追问，附面经整理和打印总册。
- 笔记分类、搜索、章节导航、公式和图解、Markdown 下载、打印。
- 编辑草稿只保存在读者当前浏览器中，不会修改公开网站。

这是 GitHub Pages 静态版本，不需要服务器或数据库。在线评论暂未开放；旧平台的评论数据库没有包含在本仓库中，也没有被删除。更换网址后，旧网址下的浏览器草稿不会自动转移，迁移前应在旧页面下载修改稿。

## 更新与发布

直接编辑 `notes.md`、`inference-infra.md` 和相应资源。公式变更时，同步更新 `assets/formulas.json` 或 `assets/infra-formulas.json`；笔记目录位于 `assets/notebook.json`。

DeepInfer 正文为 `deepinfer.md`，第 3、5 章卡片分别进入两套动画，第 6 章链接到 DSpark 调度部分。DSpark 页面为 `notes/deepinfer-dspark.html`，源码在 `assets/deepinfer-dspark/`：`model.mjs` 管理教学数值，`player.mjs` / `player.css` 管理放映，`draft-diagram.mjs` 绘制草稿流程图。修改后运行 `node scripts/check-deepinfer-dspark.mjs`，并按 [HANDOFF.md](HANDOFF.md) 检查桌面与手机交互。

### 基础与面试资料

`study/` 是 [ai-study-notes](https://github.com/pyq623/ai-study-notes) 的公开构建产物，原仓库继续作为这些讲义与题目的编辑来源。导入版本记录在 `study/import-source.json`。本站调整了标题、导航和迁移说明，保留原题目、题号、公式、图示、判分、口述笔记和导出功能。

更新时，先在原仓库按照其 README 安装依赖，运行 `npm run build && npm test`，再在本仓库运行：

```sh
python3 scripts/import-study.py /absolute/path/to/ai-study-notes
```

核对首页 `assets/notebook.json` 的介绍与题数后提交发布。新页面入口为 `/study/study.html` 和 `/study/interview.html`；原 GitHub Pages 网址仍保留。浏览器按域名隔离存储，旧站答题记录和口述笔记不会自动转移；请在旧站导出备份，当前练习不支持导入。

GitHub Pages 从 `main` 分支的根目录发布。`.nojekyll` 关闭 Jekyll 处理，不需要安装依赖或执行构建命令。

本地预览：

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

打开 http://127.0.0.1:8000/ 。笔记阅读页位于 `notes/`。站内使用相对路径，兼容 GitHub 项目网址和后续自定义域名。

`CNAME` 声明自定义域名 `sideinn.ink`。阿里云 DNS 的 `@` A 记录指向 GitHub Pages 的四个地址：`185.199.108.153`、`185.199.109.153`、`185.199.110.153`、`185.199.111.153`。HTTPS 证书由 GitHub Pages 免费签发和维护。

## 第三方资源

Marked 和 MathJax 的许可随文件保存在 `assets/vendor/`。引用的 SGLang 官方图出处见 `assets/infra-diagrams/SOURCES.md`；论文和资料来源保留在各篇笔记中。第三方资源遵循各自许可，其他内容未在此授予额外许可。

`study/` 导入资料由 pyq623 整理，其代码与样式采用 MIT，原创内容采用 CC BY 4.0；具体范围见该目录的 `LICENSE`、`LICENSE-CONTENT.md` 和 `THIRD_PARTY_NOTICES.md`。KaTeX 许可保留在 `study/assets/katex/LICENSE`。这些许可不扩展到本站其他内容。
