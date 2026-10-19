# 支线客栈

收留好奇心，记录未完成的探索。关于强化学习、智能体与推理系统的中文笔记。

免费测试网址：https://pyq623.github.io/sideinn/

## 内容与功能

- RSI 递归自我改进：13 章、18 个公式。
- vLLM、SGLang 与推理基础设施：18 章、35 个公式。
- 笔记分类、搜索、章节导航、公式和图解、Markdown 下载、打印。
- 编辑草稿只保存在读者当前浏览器中，不会修改公开网站。

这是 GitHub Pages 静态版本，不需要服务器或数据库。在线评论暂未开放；旧平台的评论数据库没有包含在本仓库中，也没有被删除。更换网址后，旧网址下的浏览器草稿不会自动转移，迁移前应在旧页面下载修改稿。

## 更新与发布

直接编辑 `notes.md`、`inference-infra.md` 和相应资源。公式变更时，同步更新 `assets/formulas.json` 或 `assets/infra-formulas.json`；笔记目录位于 `assets/notebook.json`。

GitHub Pages 从 `main` 分支的根目录发布。`.nojekyll` 关闭 Jekyll 处理，不需要安装依赖或执行构建命令。

本地预览：

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

打开 http://127.0.0.1:8000/ 。笔记阅读页位于 `notes/`。站内使用相对路径，兼容 GitHub 项目网址和后续自定义域名。

当前尚未绑定 `sideinn.ink`。先验证目标读者的网络能打开免费测试网址，再在 GitHub Pages 设置中绑定域名并更新 DNS。不要在尚未准备好时添加 `CNAME`，以免测试网址提前跳转到原站。

## 第三方资源

Marked 和 MathJax 的许可随文件保存在 `assets/vendor/`。引用的 SGLang 官方图出处见 `assets/infra-diagrams/SOURCES.md`；论文和资料来源保留在各篇笔记中。第三方资源遵循各自许可，其他内容未在此授予额外许可。
