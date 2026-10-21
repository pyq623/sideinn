#!/usr/bin/env python3
"""Import the user's Hot100 DOCX as static, directly linked review pages.

Usage: python scripts/import-hot100.py /path/to/Hot100算法题复习手册.docx
Requires lxml and Pillow. Original document is read-only; output is hot100/.
"""
import argparse
import hashlib
import html
import json
import re
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as ET
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
W = '{' + NS['w'] + '}'
esc = html.escape


def plain(node):
    parts = []
    for x in node.iter():
        if x.tag == W + 't':
            parts.append(x.text or '')
        elif x.tag == W + 'br' and x.get(W + 'type') not in ('page', 'column'):
            parts.append('\n')
        elif x.tag == W + 'tab':
            parts.append('\t')
    return re.sub(r'\s*DOCMARK\w+', '', ''.join(parts))


def style(node):
    return next(iter(node.xpath('./w:pPr/w:pStyle/@w:val', namespaces=NS)), '')


def shell(title, body, page_id=''):
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} · 支线客栈</title><meta name="description" content="Hot100 算法复习：题面、思路、Python 代码、易错点与技巧总结。按专题和题号直接定位。">
<link rel="stylesheet" href="../theme.css?v=20261019"><link rel="stylesheet" href="./hot100.css?v=20261021-painted-logo"><script defer src="./hot100.js?v=20261021"></script></head>
<body data-problem-id="{esc(page_id)}"><a class="skip" href="#main">跳到正文</a>
<header class="top"><a class="brand" href="../"><img class="brand-logo" src="../assets/sideinn-logo-painted-v2-256.png" width="54" height="54" alt=""><span>支线客栈</span></a><nav><a href="../#notebooks">所有笔记</a><a href="./">Hot100 目录</a></nav></header>
{body}<footer class="footer">支线客栈 · 算法与 Hot100 <span>复习标记仅保存在当前浏览器</span></footer></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    out = ROOT / 'hot100'
    (out / 'images').mkdir(parents=True, exist_ok=True)
    archive = ZipFile(args.source)
    doc = ET.fromstring(archive.read('word/document.xml'))
    rels = {x.get('Id'): x.get('Target') for x in ET.fromstring(archive.read('word/_rels/document.xml.rels'))}
    blocks = list(doc.find('w:body', NS))
    first = next(i for i, p in enumerate(blocks) if style(p) == '3')
    entries, categories, current, category = [], [], None, None
    for block in blocks[first:]:
        s, text = style(block), plain(block).strip()
        if s == '3':
            category = {'id': f'topic-{len(categories)+1}', 'title': re.sub(r'^第[一二三四五六七八九十]+章\s*', '', text), 'intro': []}
            categories.append(category)
            current = None
        elif s == '4':
            current = {'title': text, 'category': category['title'], 'categoryId': category['id'], 'blocks': [], 'appendix': text.startswith('附录')}
            entries.append(current)
        elif current is not None:
            current['blocks'].append(block)
        elif text:
            category['intro'].append(text)

    for i, entry in enumerate(entries):
        subtitle = next((plain(p).strip() for p in entry['blocks'] if style(p) == '170'), '')
        match = re.search(r'LeetCode\s+(\d+)', subtitle)
        entry['leetcode'] = match.group(1) if match else ''
        entry['id'] = 'lc-' + entry['leetcode'] if match else f'tip-{sum(e["appendix"] for e in entries[:i+1])}'
        entry['href'] = entry['id'] + '.html'
        entry['subtitle'] = subtitle
        meta = next((plain(p).strip() for p in entry['blocks'] if style(p) == '169'), '')
        entry['difficulty'] = next(iter(re.findall(r'难度\s+(简单|中等|困难)', meta)), '技巧')
        entry['search'] = ' '.join([entry['title'], subtitle, entry['category'], meta])

    assert len(entries) == 100 and sum(not e['appendix'] for e in entries) == 96
    assert len({e['id'] for e in entries}) == len(entries)
    bookmark_urls = {}
    for entry in entries:
        # Source heading bookmark names are used by Word's internal links.
        heading = next(b for b in blocks if style(b) == '4' and plain(b).strip() == entry['title'])
        for name in heading.xpath('.//w:bookmarkStart/@w:name', namespaces=NS):
            bookmark_urls[name] = './' + entry['href']

    media, image_occurrences, code_count, table_count = set(), 0, 0, 0

    def image_markup(node):
        nonlocal image_occurrences
        results = []
        for blip in node.findall('.//a:blip', NS):
            target = rels[blip.get('{' + NS['r'] + '}embed')]
            name = Path(target).name
            data = archive.read('word/' + target)
            (out / 'images' / name).write_bytes(data)
            media.add(name)
            image_occurrences += 1
            with Image.open(BytesIO(data)) as im:
                width, height = im.size
            results.append(f'<figure><a href="./images/{esc(name)}" target="_blank" rel="noopener" aria-label="查看原图"><img src="./images/{esc(name)}" width="{width}" height="{height}" loading="lazy" alt="{esc(active_title)} · 原笔记图示"></a><figcaption>原笔记图示 · 点击查看原图</figcaption></figure>')
        return ''.join(results)

    def inline(node):
        result = []
        field_url, field_open = '', False
        for child in node:
            tag = ET.QName(child).localname
            if tag == 'r':
                instruction = ''.join(child.xpath('.//w:instrText/text()', namespaces=NS))
                match = re.search(r'HYPERLINK\s+"([^\"]+)"', instruction)
                if match and match.group(1).startswith(('https://', 'http://')):
                    field_url = match.group(1)
                field = child.find('w:fldChar', NS)
                if field is not None:
                    if field.get(W+'fldCharType') == 'separate' and field_url:
                        result.append(f'<a href="{esc(field_url)}" target="_blank" rel="noopener noreferrer">')
                        field_open = True
                    elif field.get(W+'fldCharType') == 'end':
                        if field_open:
                            result.append('</a>')
                        field_url, field_open = '', False
                text = esc(plain(child)).replace('\n', '<br>')
                pr = child.find('w:rPr', NS)
                if pr is not None:
                    for key, wrapper in [('b', 'strong'), ('i', 'em')]:
                        prop = pr.find('w:' + key, NS)
                        if prop is not None and prop.get(W+'val') not in ('0', 'false'):
                            text = f'<{wrapper}>{text}</{wrapper}>'
                    vert = pr.find('w:vertAlign', NS)
                    if vert is not None and vert.get(W+'val') in ('superscript', 'subscript'):
                        wrapper = 'sup' if vert.get(W+'val') == 'superscript' else 'sub'
                        text = f'<{wrapper}>{text}</{wrapper}>'
                    fonts = pr.find('w:rFonts', NS)
                    if fonts is not None and fonts.get(W+'ascii') in ('Menlo', 'Consolas', 'Courier New'):
                        text = f'<code>{text}</code>'
                result.append(text)
            elif tag == 'hyperlink':
                url = rels.get(child.get('{' + NS['r'] + '}id'), '')
                url = bookmark_urls.get(child.get(W+'anchor'), url)
                label = inline(child)
                if url.startswith(('https://', 'http://', './')):
                    result.append(f'<a href="{esc(url)}"'+(' target="_blank" rel="noopener noreferrer"' if url.startswith('http') else '')+f'>{label}</a>')
                else:
                    result.append(label)
            elif tag not in ('pPr', 'bookmarkStart', 'bookmarkEnd'):
                result.append(inline(child))
        return ''.join(result)

    headings = []

    def render(p):
        nonlocal code_count, table_count
        tag = ET.QName(p).localname
        if tag == 'sectPr':
            return ''
        if tag == 'tbl':
            table_count += 1
            rows = p.findall('w:tr', NS)
            # Word also stores some code blocks in single-cell tables.
            if len(rows) == 1 and len(rows[0].findall('w:tc', NS)) == 1:
                return ''.join(render(x) for x in rows[0].find('w:tc', NS) if ET.QName(x).localname in ('p', 'tbl'))
            content = []
            for idx, row in enumerate(rows):
                cells = []
                for cell in row.findall('w:tc', NS):
                    cell_tag = 'th' if idx == 0 else 'td'
                    cells.append(f'<{cell_tag}'+(' scope="col"' if idx == 0 else '')+'>'+''.join(render(x) for x in cell if ET.QName(x).localname in ('p', 'tbl'))+f'</{cell_tag}>')
                content.append('<tr>'+''.join(cells)+'</tr>')
            return '<div class="table-scroll" tabindex="0" aria-label="可横向滚动的表格"><table>'+''.join(content)+'</table></div>'
        if tag != 'p':
            raise ValueError('Unhandled block: ' + tag)
        text, s = plain(p), style(p)
        pictures = image_markup(p)
        if not text.strip():
            return pictures
        if s == '171':
            code_count += 1
            return '<pre tabindex="0"><code>'+esc(text)+'</code></pre>'+pictures
        if s in ('5', '6'):
            hid = f'section-{len(headings)+1}'
            headings.append((hid, text.strip(), s))
            level = 'h2' if s == '5' else 'h3'
            return f'<{level} id="{hid}">{inline(p)}</{level}>'+pictures
        if s in ('170', '169'):
            return f'<p class="metadata">{inline(p)}</p>'+pictures
        if s == '172':
            return f'<p class="caption">{inline(p)}</p>'+pictures
        if s in ('16', '18', '23', '13', '14', '20'):
            ordered = s in ('13', '14', '20')
            return ('<ol>' if ordered else '<ul>')+'<li>'+inline(p)+'</li>'+('</ol>' if ordered else '</ul>')+pictures
        return '<p>'+inline(p)+'</p>'+pictures

    for i, entry in enumerate(entries):
        active_title, headings = entry['title'], []
        content = ''.join(render(b) for b in entry['blocks'])
        # Adjacent source list paragraphs belong to the same list.
        content = content.replace('</ul><ul>', '').replace('</ol><ol>', '')
        content = content.replace('</code></pre><pre tabindex="0"><code>', '\n')
        toc = ''.join(f'<a class="level-{s}" href="#{hid}">{esc(t)}</a>' for hid, t, s in headings)
        nav = '<nav class="neighbors" aria-label="前后篇">'
        for j, label in [(i-1, '← 上一篇'), (i+1, '下一篇 →')]:
            if 0 <= j < len(entries):
                other = entries[j]
                nav += f'<a href="./{other["href"]}"><small>{label}</small>{esc(other["title"])}</a>'
        nav += '</nav>'
        controls = '' if entry['appendix'] else '''<div class="review-controls"><label for="review-status">复习状态</label><select id="review-status"><option value="new">未复习</option><option value="review">待巩固</option><option value="done">已掌握</option></select><span id="save-state" role="status"></span></div>'''
        body = f'''<div class="reading"><aside class="toc"><a href="./#{entry['categoryId']}">← {esc(entry['category'])}目录</a><details open><summary>本题导航</summary><nav>{toc}</nav></details></aside>
<main id="main" class="paper"><div class="eyebrow">HOT100 / {esc(entry['category'])}</div><h1>{esc(entry['title'])}</h1>{controls}<article>{content}</article>{nav}</main></div>'''
        (out / entry['href']).write_text(shell(entry['title'], body, entry['id']), encoding='utf-8')

    sections = []
    for category in categories:
        items = [e for e in entries if e['categoryId'] == category['id']]
        cards = []
        for e in items:
            cards.append(f'''<li data-search="{esc(e['search'])}" data-id="{e['id']}" data-topic="{category['id']}" data-kind="{'tip' if e['appendix'] else 'problem'}"><a href="./{e['href']}"><span class="problem-title">{esc(e['title'])}</span><span class="problem-meta">{('LeetCode '+e['leetcode']+' · ') if e['leetcode'] else ''}{e['difficulty']} <span class="status" data-status-for="{e['id']}"></span></span></a></li>''')
        intro = ''.join('<p class="topic-intro">'+esc(t)+'</p>' for t in category['intro'])
        sections.append(f'<section class="topic" id="{category["id"]}"><h2>{esc(category["title"])} <span>{len(items)} '+('篇' if items[0]['appendix'] else '题')+'</span></h2>'+intro+'<ul class="problem-list">'+''.join(cards)+'</ul></section>')
    options = ''.join(f'<option value="{c["id"]}">{esc(c["title"])}</option>' for c in categories)
    topic_links = ''.join(f'<a href="#{c["id"]}">{esc(c["title"])}</a>' for c in categories)
    body = f'''<main id="main" class="catalog"><section class="catalog-intro"><div class="eyebrow">ALGORITHMS / PYTHON REVIEW</div><h1>Hot100 算法复习</h1><p>先独立想解法，再对照代码和易错点。沿用原手册的专题与题目顺序，点击题名即可进入正文。</p><div class="stats"><span><strong>96</strong> 道题</span><span><strong>16</strong> 个专题</span><span><strong>4</strong> 篇技巧总结</span></div><p class="source-note">含第 31 题前的补充题「反转链表 II」。网页目录直接对应正文，不使用原手册的纸面页码。</p></section>
<div class="filters"><label class="search-label">查找题目<input id="question-search" type="search" placeholder="题名、LeetCode 题号或关键词" autocomplete="off"></label><label>专题<select id="topic-filter"><option value="">全部专题</option>{options}</select></label><label>复习状态<select id="status-filter"><option value="">全部状态</option><option value="new">未复习</option><option value="review">待巩固</option><option value="done">已掌握</option></select></label></div>
<div class="catalog-status"><span id="result-count" role="status">96 道题 · 4 篇技巧总结</span><span id="review-count"></span></div><nav class="topic-links" aria-label="专题直达">{topic_links}</nav><div id="questions">{''.join(sections)}</div><div id="empty" hidden><p>没有匹配的题目，试试其他关键词或专题。</p><button id="clear-filters">清空筛选</button></div><noscript>目录和题目均可直接阅读；搜索与复习标记需要启用 JavaScript。</noscript></main>'''
    (out / 'index.html').write_text(shell('Hot100 算法复习', body), encoding='utf-8')
    manifest = {'source': args.source.name, 'sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(), 'problems': 96, 'topics': 16, 'tips': 4, 'images': len(media), 'imageOccurrences': image_occurrences, 'codeBlocks': code_count, 'sourceTables': table_count, 'entries': [{k:v for k,v in e.items() if k not in ('blocks','search')} for e in entries]}
    (out / 'import-source.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='entries'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
