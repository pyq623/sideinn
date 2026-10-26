#!/usr/bin/env python3
"""Import four user attention implementations; preserve .py files verbatim.
Usage: python3 scripts/import-attention.py /path/to/2-attention-handwrite
"""
import argparse
import hashlib
import html
import io
import json
import keyword
from pathlib import Path
import tokenize

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = [
    dict(id='mha', name='MHA', full='Multi-Head Attention', title='多头注意力',
         source='1-mha/mha_practice.py', cls='MultiHeadAttention',
         description='Q、K、V 一次投影，再拆成多个头。重点复习 reshape、transpose、缩放点积与 mask。',
         note='输入和输出均为 [B, S, D]；D 必须能被头数 H 整除。',
         constructor='MultiHeadAttention(d_model=16, num_heads=4)'),
    dict(id='mqa', name='MQA', full='Multi-Query Attention', title='多查询注意力',
         source='2-mqa/mqa_practice.py', cls='MultiQueryAttention',
         description='多个 Q 头共享同一对 K/V。重点复习 KV 投影、view / unbind 与头维度广播。',
         note='Q 为 [B, H, S, head_dim]，K/V 为 [B, 1, S, head_dim]；矩阵乘法时沿头维度广播，相当于 GQA 的 group=1。',
         constructor='MultiQueryAttention(d_model=16, num_heads=4)'),
    dict(id='gqa', name='GQA', full='Grouped-Query Attention', title='分组查询注意力',
         source='3-gqa/gqa_practice.py', cls='MultiGroupAttention',
         description='保留多个 Q 头，让一组 Q 头共享一对 K/V 头。对照 MHA，关注 KV 投影维度与 repeat_interleave。',
         note='这里 group 是 K/V 头数，不是每组的 Q 头数；每组 Q 头数 = num_heads // group。',
         constructor='MultiGroupAttention(d_model=16, num_heads=4, group=2)'),
    dict(id='mla', name='MLA', full='Multi-Head Latent Attention', title='多头潜在注意力',
         source='4-mla/mla_handwrite.py', cls='MultiHeadLatentAttention',
         description='先把 KV 压缩到共享 latent，再展开内容 K/V，并组合独立的 RoPE 分支。保留 RMSNorm 与 RoPE 辅助函数。',
         note='这是显式展开 K/V 的教学实现；当前 forward 没有增量 KV cache 接口。',
         constructor='MultiHeadLatentAttention(\n    d_model=16, num_heads=4, kv_lora_rank=6,\n    qk_nope_head_dim=4, qk_rope_head_dim=2, value_head_dim=4,\n)'),
]
esc = html.escape


def highlight(source):
    """Add color without changing the code's text, indentation, or newlines."""
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    parts, cursor = [], 0
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        kind = {tokenize.COMMENT: 'comment', tokenize.STRING: 'string', tokenize.NUMBER: 'number'}.get(tok.type)
        if tok.type == tokenize.NAME and keyword.iskeyword(tok.string):
            kind = 'keyword'
        if not kind:
            continue
        start = offsets[tok.start[0]-1] + tok.start[1]
        end = offsets[tok.end[0]-1] + tok.end[1]
        parts.extend([esc(source[cursor:start]), f'<span class="tok-{kind}">{esc(source[start:end])}</span>'])
        cursor = end
    return ''.join(parts) + esc(source[cursor:])


def shell(title, body):
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} · 支线客栈</title><meta name="description" content="MHA、MQA、GQA、MLA 的 PyTorch 手写代码，保留注释，支持复制与下载。">
<link rel="stylesheet" href="../theme.css?v=20261019"><link rel="stylesheet" href="../hot100/hot100.css?v=20261021-painted-logo">
<link rel="stylesheet" href="./attention.css?v=20261021-mqa"><script defer src="./attention.js?v=20261021"></script></head>
<body><a class="skip" href="#main">跳到正文</a><header class="top"><a class="brand" href="../"><img class="brand-logo" src="../assets/sideinn-logo-painted-v2-256.png" width="54" height="54" alt=""><span>支线客栈</span></a><nav><a href="../#notebooks">所有笔记</a><a href="./">手写代码目录</a></nav></header>
{body}<footer class="footer">支线客栈 · 注意力手写代码 <span>Python / PyTorch · MHA / MQA / GQA / MLA</span></footer></body></html>'''


def code_block(source, code_id):
    return f'<pre tabindex="0" aria-label="Python 代码"><code id="{code_id}" class="language-python">{highlight(source)}</code></pre>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    out = ROOT / 'attention'
    (out / 'code').mkdir(parents=True, exist_ok=True)
    manifest, cards = [], []
    for entry in ENTRIES:
        data = (args.source / entry['source']).read_bytes()
        code = data.decode('utf-8')
        filename = Path(entry['source']).name
        (out / 'code' / filename).write_bytes(data)
        manifest.append(dict(id=entry['id'], source=entry['source'], file='code/'+filename, sha256=hashlib.sha256(data).hexdigest()))
        example = f'''import torch
from {Path(filename).stem} import {entry['cls']}

model = {entry['constructor']}.eval()
x = torch.randn(2, 5, 16)  # [B, S, D]
mask = torch.ones(2, 5, dtype=torch.bool)
y = model(x, causal=True, attention_mask=mask)
print(y.shape)  # torch.Size([2, 5, 16])
'''
        links = ''.join(f'<a href="./{e["id"]}.html" {"aria-current=page" if e["id"] == entry["id"] else ""}>{e["name"]} · {e["title"]}</a>' for e in ENTRIES)
        body = f'''<div class="reading"><aside class="toc"><a href="./">← 手写代码目录</a><nav aria-label="四个注意力实现">{links}</nav><details open><summary>本页导航</summary><nav><a href="#implementation">完整代码</a><a href="#run">最小调用</a></nav></details></aside>
<main id="main" class="paper"><div class="eyebrow">ATTENTION / PYTORCH</div><h1>{entry['name']} · {entry['title']}</h1><p class="metadata">{entry['full']}</p><p>{entry['description']}</p><p class="code-note">{entry['note']}</p>
<h2 id="implementation">完整代码</h2><div class="code-actions"><span>{filename}</span><button type="button" data-copy="implementation-code">复制代码</button><a href="./code/{filename}" download>下载 .py</a><span class="copy-status" role="status"></span></div>{code_block(code, 'implementation-code')}
<h2 id="run">最小调用</h2><p>安装 PyTorch 后，将下载的 <code>{filename}</code> 放在当前目录，再运行下面的示例。<code>attention_mask</code> 中 1 表示有效 token；<code>causal=True</code> 屏蔽未来位置。</p>{code_block(example, 'example-code')}
<p class="metadata">这四份实现的 padding mask 屏蔽的是 key 位置。调用时应保证每个参与计算的 query 至少有一个可见 key，避免整行被屏蔽后 softmax 产生 NaN。</p>
<nav class="neighbors" aria-label="其他实现">{''.join(f'<a href="./{e["id"]}.html">{e["name"]} · {e["title"]} →</a>' for e in ENTRIES if e['id'] != entry['id'])}</nav></main></div>'''
        (out / (entry['id'] + '.html')).write_text(shell(entry['name']+' 手写代码', body))
        cards.append(f'<li><a href="./{entry["id"]}.html"><span class="code-name">{entry["name"]}</span><span class="problem-title">{entry["title"]}</span><span class="problem-meta">{entry["description"]}</span><span class="read-code">阅读代码 ↗</span></a></li>')
    body = f'''<main id="main" class="catalog"><section class="catalog-intro"><div class="eyebrow">ATTENTION / HANDWRITTEN CODE</div><h1>注意力手写代码</h1><p>四份独立的 PyTorch 实现。保留原代码与注释，先看张量维度，再手写复现；需要时可直接复制或下载。</p><div class="stats"><span><strong>4</strong> 份实现</span><span>MHA · MQA · GQA · MLA</span></div></section><ul class="problem-list attention-list">{''.join(cards)}</ul><p class="source-note">整理自原「Attention 手写」笔记，每份代码均附最小调用示例。</p></main>'''
    (out / 'index.html').write_text(shell('注意力手写代码', body))
    (out / 'import-source.json').write_text(json.dumps({'source':'Formulaic/content/docs/pyq-interview/2-attention-handwrite', 'entries':manifest}, ensure_ascii=False, indent=2)+'\n')
    print('Imported MHA, MQA, GQA, MLA: 4 unchanged Python files and 5 reading pages.')


if __name__ == '__main__':
    main()
