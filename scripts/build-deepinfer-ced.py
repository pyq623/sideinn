#!/usr/bin/env python3
"""Build the CED teaching diagram from SVG shapes and pre-rendered MathJax glyphs.

Run from any directory: python3 scripts/build-deepinfer-ced.py
Formula source and self-contained MathJax SVGs live in ced-math.json.
"""
from pathlib import Path
import html
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/deepinfer-figures'
MATH = json.loads((OUT / 'ced-math.json').read_text())
ET.register_namespace('', 'http://www.w3.org/2000/svg')
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
INK, MUTED, TEAL, ORANGE = '#2c3833', '#68736c', '#23695f', '#ac592b'
PAPER, GREEN, PEACH = '#faf8f2', '#e3eee7', '#f6e7d7'
parts = []

def text(x, y, value, size=23, fill=INK, weight=400, anchor='start'):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(value)}</text>')

def box(x, y, w, h, fill, stroke='none', radius=12):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')

def arrow(path, kind='cache', dashed=False):
    color=TEAL if kind=='cache' else ORANGE
    dash=' stroke-dasharray="7 5"' if dashed else ''
    parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"{dash} marker-end="url(#{kind})"/>')

def math(key, x, y, size=29, fill=INK, center=False):
    item=MATH[key]
    svg=ET.fromstring(item['svg'])
    _,_,w,h=map(float,svg.attrib['viewBox'].split())
    # MathJax SVG uses 1000 coordinate units per em.
    width,height=w*size/1000,h*size/1000
    svg.attrib.update(x=str(x-width/2 if center else x),y=str(y),width=str(width),height=str(height),color=fill)
    svg.attrib.pop('style',None)
    svg.attrib['aria-label']=item['tex']
    # No external font files are needed: the original TeX is retained as metadata.
    parts.append(ET.tostring(svg,encoding='unicode'))

box(0,0,1000,1500,PAPER,radius=0)
text(40,55,'CED：两路数据如何进入 decoder',35,weight=650)
text(40,93,'Prefill 示意 · 输入未命中前缀缓存 · 配合 bounded replay',21,MUTED)
# Full prompt through encoder.
for x,w in [(40,220),(330,220),(620,340)]:box(x,135,w,105,GREEN)
text(150,174,'完整 prompt',25,weight=600,anchor='middle');math('prompt',150,193,27,center=True)
text(440,174,'因果 encoder',25,weight=600,anchor='middle');math('layers',440,193,25,center=True)
text(790,174,'全部位置的编码表示',25,weight=600,anchor='middle');math('hidden',790,193,28,center=True)
arrow('M260 188 H320');arrow('M550 188 H610')
# The same full hidden states feed two distinct paths.
arrow('M680 240 V270 H490 V395 H445')
arrow('M850 240 V270 H980 V405 H960','state')
text(40,322,'① 缓存路径：覆盖完整输入',25,TEAL,600)
text(560,322,'② 状态路径：只保留尾部',25,ORANGE,600)
box(40,340,405,165,GREEN,TEAL)
text(242,378,'投影，构造全局 KV',25,TEAL,600,'middle')
math('projection',242,402,31,TEAL,True)
text(242,463,'按 CSA2 规则构造 / 跨层共享',20,MUTED,anchor='middle')
text(242,490,'缓存提供可供注意力读取的内容',19,MUTED,anchor='middle')
box(560,340,400,130,PEACH,ORANGE)
math('suffix',760,370,30,ORANGE,True)
text(760,442,'取尾部位置的 encoder 输出',22,ORANGE,500,'middle')
# Decoder lane: hidden stream (orange) and cache reads (green).
box(50,550,910,590,'#fffdf8','#cfcfc0',16)
text(80,592,'decoder 内的一个代表层',26,weight=650)
math('layer',420,567,26,MUTED)
arrow('M760 470 V520 H740 V620','state')
box(560,620,360,75,PEACH,ORANGE)
text(740,665,'本层输入的隐藏状态',25,ORANGE,600,'middle')
text(80,664,'缓存与隐藏状态是两类输入',23,TEAL,600)
text(80,700,'在注意力计算处汇合',23,MUTED)
text(560,731,'由本层状态计算',20,ORANGE)
arrow('M740 695 V745 H635 V765','state')
arrow('M740 745 H845 V765','state')
box(560,765,150,90,PEACH,ORANGE)
math('query',635,778,34,ORANGE,True)
text(635,839,'查询向量',18,MUTED,anchor='middle')
box(770,765,150,90,PEACH,ORANGE)
math('local',845,778,29,ORANGE,True)
text(845,839,'局部键值',18,MUTED,anchor='middle')
box(80,805,370,175,GREEN,TEAL)
text(265,846,'注意力计算',29,TEAL,650,'middle')
text(265,891,'查询向量读取选中的全局键值',22,anchor='middle')
text(265,926,'以及本层的局部键值',23,anchor='middle')
text(265,958,'遵守因果掩码与局部窗口限制',18,MUTED,anchor='middle')
# A distinct global-KV arrow reaches the attention block, not the hidden input.
arrow('M242 505 V525 H25 V885 H80')
text(80,782,'全局缓存读取',20,TEAL,600)
arrow('M560 810 H490 V870 H450','state')
arrow('M845 855 V945 H450','state')
box(560,1000,360,80,PEACH,ORANGE)
text(740,1032,'注意力输出 → 后续层内计算',21,ORANGE,600,'middle')
text(740,1063,'更新隐藏状态，传入下一层',21,ORANGE,anchor='middle')
arrow('M265 980 V1040 H560','state')
arrow('M920 1040 H945 V657 H920','state',True)
text(80,1117,'虚线表示逐层重复；图中省略残差流、MoE、归一化及具体缓存布局。',19,MUTED)
# Boundaries and asymptotic work are deliberately outside the dataflow.
text(40,1181,'有限重放近似恢复局部 SWA 状态；更早输入的全局内容仍可被读取。',23,ORANGE,600)
box(40,1210,920,135,'#eef3ed')
text(62,1243,'简化主干层工作量',21,TEAL,600)
math('full',65,1272,28)
math('ced',425,1262,29)
text(62,1327,'不含全长 KV 投影、通信和算子差异；不是端到端耗时公式。',19,MUTED)
box(40,1365,920,65,GREEN)
text(62,1405,'DECODE',21,TEAL,650)
text(218,1405,'新位置 → encoder → decoder（读取 / 更新缓存）→ 采样',23,INK,500)
math('n',40,1450,23,MUTED);text(70,1469,'输入 token 数',20,MUTED)
math('depth',300,1450,23,MUTED);text(330,1469,'总层数',20,MUTED)
math('window',470,1450,23,MUTED);text(505,1469,'尾部重放 token 数',20,MUTED)
text(960,1484,'教学示意 · 非官方架构原图',17,MUTED,anchor='end')
svg='''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1000" height="1500" viewBox="0 0 1000 1500" role="img" aria-labelledby="title desc">
<title id="title">CED 配合有限重放：全局 KV 与尾部隐藏状态的两路输入</title>
<desc id="desc">全部 encoder 输出用于投影构造全局 KV；仅尾部编码表示继续进入 decoder。每层从当前状态生成 Q 和局部 SWA KV，注意力读取选中的全局 KV 与局部 KV，再更新隐藏状态。有限重放为近似恢复。数学符号由 MathJax 将 LaTeX 排版为矢量路径。</desc>
<defs><marker id="cache" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#23695f"/></marker><marker id="state" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#ac592b"/></marker></defs>
<g font-family="-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif">'''+''.join(parts)+'</g></svg>\n'
ET.fromstring(svg)
(OUT/'02-ced-latex.svg').write_text(svg)
print('Built',OUT/'02-ced-latex.svg')
