#!/usr/bin/env python3
"""Import a built ai-study-notes checkout, preserving content and exercise IDs.

Usage: python3 scripts/import-study.py /path/to/ai-study-notes
Build and test that checkout first. Only its public docs output is imported.
"""
from pathlib import Path
import argparse
import json
import re
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkout', type=Path)
args = parser.parse_args()
source = args.checkout.resolve()
docs = source / 'docs'
destination = Path(__file__).resolve().parents[1] / 'study'
required = ['index.html', 'study.html', 'interview.html', 'interview-unit-1.html']
for name in required:
    if not (docs / name).is_file():
        raise SystemExit('Missing built page: ' + str(docs / name))
revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
destination.mkdir(exist_ok=True)
shutil.copytree(docs, destination, dirs_exist_ok=True)
old_brand = '<a class="brand" href="index.html">AI学习与面试笔记</a>'
new_brand = '<a class="brand" href="../" aria-label="返回支线客栈笔记目录"><img class="brand-logo" src="../assets/sideinn-logo-painted-v2-256.png" width="54" height="54" alt=""><span>支线客栈<small>论文 · 实验 · 学习连载</small></span></a>'
nav = '<nav class="site-nav" aria-label="全站导航">'
notice = '<p>本专题由 pyq623 整理，合并至支线客栈时统一了导航与阅读样式，讲义与题目保持原样。旧网址的答题与口述记录不会自动转移，请在旧站导出留存。</p>'
for page in destination.glob('*.html'):
    content = page.read_text()
    if page.name.startswith('coding-'):
        content = content.replace('href="https://sideinn.ink/', 'href="../')
    if old_brand not in content or nav not in content:
        raise SystemExit('Source navigation changed; review importer: ' + page.name)
    content = content.replace(old_brand, new_brand, 1)
    content = content.replace(nav, nav + '<a href="../">← 支线首页</a>', 1)
    content = content.replace(' · AI学习与面试笔记</title>', ' · 支线客栈</title>')
    content = content.replace('<footer class="site-footer">', '<footer class="site-footer">' + notice, 1)
    # The source 404 page has an absolute base for its former project URL.
    content = content.replace('href="/ai-study-notes/"', 'href="/study/"')
    # Site-only reader shell: keep source prose, exercise IDs, scripts and storage intact.
    content = content.replace('</head>', '<link rel="stylesheet" href="../theme.css?v=20261019"><link rel="stylesheet" href="../study-reader.css?v=20261021"><script defer src="../study-reader.js?v=20261021"></script></head>', 1)
    header = re.search(r'<header class="site-header">.*?</header>', content, re.S)
    chapter = re.search(r'<nav class="site-nav chapter-nav".*?</nav>', header[0], re.S)
    chapter_nav = chapter[0] if chapter else ''
    clean_header = header[0].replace(chapter_nav, '')
    clean_header = clean_header.replace('</header>', '<button type="button" class="reader-menu" aria-expanded="false" aria-controls="reader-toc">目录</button></header>')
    content = content.replace('<div class="wrap">' + header[0], clean_header + '<div class="wrap">', 1)
    switcher = '<details class="reader-chapters"><summary>切换章节</summary>' + chapter_nav + '</details>' if chapter_nav else ''
    toc_intro = '<a class="reader-back" href="../#notebooks">← 返回笔记目录</a><p class="reader-label">阅读目录</p>'
    breadcrumbs = '<div class="reader-meta">支线客栈 / ' + ('面试手撕专项复习' if 'class="area-coding"' in content else '面试与项目表达' if 'class="area-interview"' in content else '基础与笔试') + '</div>'
    if '<div class="course-layout">' in content:
        # Interview title/controls originally sit above the two-column reader.
        opening = re.search(r'<div class="wrap">(.*?)<div class="course-layout">', content, re.S)
        intro = opening[1]
        content = content.replace(opening[0], '<div class="wrap"><div class="course-layout">', 1)
        content = content.replace('<aside class="course-toc">', '<aside class="course-toc" id="reader-toc" aria-label="阅读目录">' + toc_intro, 1)
        content = content.replace('</aside>', switcher + '</aside>', 1)
        content = content.replace('<main id="main-content" class="course-article">', '<main id="main-content" class="course-article">' + breadcrumbs + intro, 1)
    else:
        # Collection, sources and print pages share the same shell without fabricating a TOC.
        links = chapter_nav or '<nav class="site-nav chapter-nav"><a href="study.html">基础与工程笔记</a><a href="interview.html">面试笔记</a><a href="interview-research.html">面经与追问</a><a href="about.html">关于与许可</a></nav>'
        if 'class="area-coding"' in content:
            links = '<nav class="site-nav chapter-nav"><a href="coding-practice.html">手撕专项目录</a><a href="coding-ai-infra.html">AI Infra</a></nav>'
        sidebar = '<aside class="course-toc" id="reader-toc" aria-label="阅读目录">' + toc_intro + '<details open><summary>专题导航</summary>' + links + '</details></aside>'
        content = content.replace('<div class="wrap">', '<div class="wrap"><div class="course-layout reader-overview">' + sidebar, 1)
        content = content.replace('</main>', '</main></div>', 1)
    content = content.replace('</body>', '<button class="reader-backdrop" type="button" aria-label="关闭目录" hidden></button></body>', 1)
    page.write_text(content)
for name in ('LICENSE', 'LICENSE-CONTENT.md', 'THIRD_PARTY_NOTICES.md'):
    shutil.copy2(source / name, destination / name)
(destination / 'import-source.json').write_text(json.dumps({
    'repository': 'https://github.com/pyq623/ai-study-notes',
    'commit': revision,
    'changes': 'Sideinn reader shell, branding, navigation, migration notice and 404 base path; educational content, exercise IDs and storage scripts preserved.',
}, ensure_ascii=False, indent=2) + '\n')
print('Imported study notes at ' + revision)
