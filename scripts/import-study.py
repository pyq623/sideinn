#!/usr/bin/env python3
"""Import a built ai-study-notes checkout, preserving content and exercise IDs.

Usage: python3 scripts/import-study.py /path/to/ai-study-notes
Build and test that checkout first. Only its public docs output is imported.
"""
from pathlib import Path
import argparse
import json
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
new_brand = '<a class="brand" href="../">支线客栈</a><p class="meta">AI学习 · 笔试练习 · 面试复习</p>'
nav = '<nav class="site-nav" aria-label="全站导航">'
notice = '<p>本专题由 pyq623 整理，合并至支线客栈时调整了站点导航，讲义与题目保持原样。旧网址的答题与口述记录不会自动转移，请在旧站导出留存。</p>'
for page in destination.glob('*.html'):
    content = page.read_text()
    if old_brand not in content or nav not in content:
        raise SystemExit('Source navigation changed; review importer: ' + page.name)
    content = content.replace(old_brand, new_brand, 1)
    content = content.replace(nav, nav + '<a href="../">← 支线首页</a>', 1)
    content = content.replace(' · AI学习与面试笔记</title>', ' · 支线客栈</title>')
    content = content.replace('<footer class="site-footer">', '<footer class="site-footer">' + notice, 1)
    # The source 404 page has an absolute base for its former project URL.
    content = content.replace('href="/ai-study-notes/"', 'href="/study/"')
    page.write_text(content)
for name in ('LICENSE', 'LICENSE-CONTENT.md', 'THIRD_PARTY_NOTICES.md'):
    shutil.copy2(source / name, destination / name)
(destination / 'import-source.json').write_text(json.dumps({
    'repository': 'https://github.com/pyq623/ai-study-notes',
    'commit': revision,
    'changes': 'Site title, navigation, migration notice and 404 base path only; educational content and exercise IDs preserved.',
}, ensure_ascii=False, indent=2) + '\n')
print('Imported study notes at ' + revision)
