#!/usr/bin/env python3
"""Build the six original, static teaching diagrams for the V4.1-Flash note.
No dependency or network access. Paper figures are kept separately, unmodified.
"""
from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parents[1] / 'assets/v41-flash-figures'
INK, MUTED, TEAL, GOLD = '#203b35', '#61716a', '#22786b', '#a46b2a'
PALE, SAND, BG, LINE = '#e5f1eb', '#f7ecd8', '#faf8f2', '#cfdbd2'

class Figure:
    def __init__(self, title, subtitle, height=760):
        self.height = height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" role="img" aria-labelledby="title desc">', f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>', '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#61716a"/></marker></defs>', f'<rect width="1000" height="{height}" rx="22" fill="{BG}"/>', '<g font-family="PingFang SC, Noto Sans CJK SC, Microsoft YaHei, sans-serif">']
        self.text(40, 50, title, 29, INK, '650')
        self.text(40, 86, subtitle, 18, MUTED)
    def text(self, x, y, value, size=21, color=INK, weight='400', anchor='start'):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>')
    def box(self, x,y,w,h, title, lines=(), fill=PALE, color=TEAL):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{LINE}"/>')
        self.text(x+20,y+36,title,23,color,'650')
        for i,line in enumerate(lines): self.text(x+20,y+73+i*31,line,20)
    def arrow(self, points):
        self.parts.append(f'<polyline points="{points}" fill="none" stroke="{MUTED}" stroke-width="2.5" stroke-linejoin="round" marker-end="url(#arrow)"/>')
    def line(self, x1,y1,x2,y2):
        self.parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{LINE}" stroke-width="2"/>')
    def save(self, name):
        self.text(40,self.height-23,'支线客栈 · 原创教学示意 · 依据技术报告 v1，2026-09-17',15,MUTED)
        OUT.joinpath(name).write_text('\n'.join(self.parts+['</g></svg>']))

f=Figure('01  长任务的三本成本账','同一条 Agent 请求链，分别考察计算、容量和恢复。',720)
for x,n,title,lines in [
 (40,'01','新输入的计算',['长历史要走多少层？','CED：全长经过 encoder','尾部窗口经过 decoder']),
 (355,'02','历史的容量',['保存几份、每份多大？','CSA2：序列与跨层共享','FP4：数据 + 分组 scale']),
 (670,'03','检索与恢复',['搜索、搬运、重放什么？','分层索引：缩小候选池','SWA：只恢复有限尾部'])]:
 f.box(x,135,290,234,title,lines)
 f.text(x+20,342,n,30,TEAL,'650')
f.box(40,405,920,105,'生成阶段：新 token 仍走完整 40 层',['DSpark 另行优化串行生成；mHC 另行优化残差混合的激活访存。'],SAND,GOLD)
f.box(40,540,920,112,'读到“减少”时，先补齐分母',['Global KV 容量 × 1/4 ≠ 持久 KV 容量 × 1/8 ≠ 整机显存 × 1/8。'], '#fff',INK)
f.save('01-cost-map.svg')

f=Figure('02  CED：全局缓存与隐藏状态分流','Prefill 的简化路径；两个 20 层模块仍属于同一个因果模型。',940)
f.box(310,125,380,83,'输入 N 个 token',[])
f.arrow('500,208 500,235')
f.box(310,240,380,110,'Encoder · 20 层',['全 N 个位置 → 最终表示 H'])
f.arrow('380,350 380,390 245,390 245,425')
f.arrow('620,350 620,390 755,390 755,425')
f.text(50,401,'全部位置的 H',18,TEAL)
f.text(785,401,'尾部 W 个 H',18,GOLD)
f.box(40,430,410,140,'Global KV 投影与缓存',['由 H 投影 C、Z','各 decoder 层共享此历史内容'])
f.box(550,430,410,140,'Decoder · 20 层',['尾部隐藏状态逐层前向','每层自己的 Q、SWA、MoE'],SAND,GOLD)
f.arrow('450,502 543,502')
f.text(479,488,'读',17,MUTED)
f.arrow('755,570 755,605')
f.box(550,610,410,93,'尾部局部状态与输出',[],SAND,GOLD)
f.box(40,610,410,93,'全长历史已可检索',[])
f.box(40,745,920,126,'边界：尾部重放是经过训练适配的近似',['历史不必全跑 decoder；生成时，每个新 token 仍走 encoder + decoder。','W = 128；示意省略常规残差、位置编码和具体压缩算子。'],'#fff',INK)
f.save('02-ced-paths.svg')

f=Figure('03  候选池不是第一次 Top-K 的同义词','人工小例子：16 个位置，块长 4，保留 2 块，每次选择 Top-2。',800)
scores=[.1,.8,.2,.3,.4,.2,.9,.1,.2,.3,.1,.4,.1,.7,.2,.3]
for b in range(4):
 x=40+b*235
 f.box(x,130,215,220,f'块 {"ABCD"[b]} · max {max(scores[b*4:b*4+4]):.1f}',[],PALE if b<2 else '#eeefea')
 for k in range(4):
  i=b*4+k
  f.text(x+20,210+k*32,f'位置 {i+1:>2}  →  {scores[i]:.1f}',19,TEAL if i in [1,6] else INK,'650' if i in [1,6] else '400')
f.box(40,390,440,115,'第一次 Full 的 Top-2',['7、2 → 用于这次注意力'],SAND,GOLD)
f.box(520,390,440,115,'候选池：按块最大分选 B、A',['池内位置：1–8；不含位置 14'])
f.arrow('740,505 740,545')
f.box(520,550,440,115,'后续 Reindex 用自己的 Q',['例如在 1–8 中重新选：3、6'])
f.text(40,578,'两种选择服务不同目的：',23,INK,'650')
f.text(40,618,'本层最终读取 ≠ 后续可搜索范围',20,MUTED)
f.text(40,712,'真实配置：Top-512；8 位置 / 块；最多 2048 块，即 16384 个候选位置。',20)
f.save('03-candidate-pool.svg')

f=Figure('04  从 FP4 位宽算到 890 B / token','按论文最终 global KV 格式计算；包括分组 scale，不包括 SWA。',790)
f.box(40,130,440,160,'Main KV · 512 维',['数据：512 × 4 / 8 = 256 B','Scale：512 / 16 × 1 = 32 B','合计：288 B / 条'])
f.box(520,130,440,160,'Indexer K · 128 维',['数据：128 × 4 / 8 = 64 B','Scale：128 / 32 × 1 = 4 B','合计：68 B / 条'])
f.arrow('260,290 260,325 500,325 500,355')
f.arrow('740,290 740,325 500,325')
f.box(235,360,530,94,'一个 global 条目：288 + 68 = 356 B',[])
f.arrow('500,454 500,490')
f.box(40,495,920,122,'每个原始 token 对应多少 global 条目？',['Encoder：3 套 × 1/2 ；Decoder：1 套 × 1 → 合计 2.5 条 / token'])
f.text(500,671,'356 × 2.5 = 890 B / 原始 token',32,TEAL,'700','middle')
f.text(500,714,'不含权重、Engram、SWA、索引地址、分配碎片和运行工作区。',19,MUTED,'400','middle')
f.save('04-byte-ledger.svg')

f=Figure('05  有限重放：保留全局历史，补回局部尾部','人工小例子 W = 4；旧前缀有 12 个位置，新输入为 13、14。',820)
f.text(40,133,'旧 Global KV 已命中，1–12 保持原值',22,TEAL,'650')
for i in range(14):
 x=40+i*66
 f.parts.append(f'<rect x="{x}" y="155" width="58" height="54" rx="8" fill="{SAND if i>=12 else PALE}" stroke="{LINE}"/>')
 f.text(x+29,190,i+1,22,INK,'500','middle')
f.text(40,245,'1–8：无须重放',20,MUTED)
f.text(568,245,'9–12：重放段',20,TEAL,'650')
f.text(832,245,'新增',20,GOLD,'650')
f.box(40,280,440,310,'重放起点 s = 9',['位置 9   可见：9','位置 10 可见：9、10','位置 11 可见：9、10、11','位置 12 可见：9、10、11、12','真实 W = 128；此处只为便于手算。'])
f.box(520,280,440,140,'重放恢复 SWA',['旧 global KV 不被重算值覆盖','局部可见范围受到起点 s 限制'])
f.box(520,450,440,140,'新增后缀 13、14',['正常追加 global KV','正常滚动更新局部 SWA'],SAND,GOLD)
f.box(40,630,920,120,'近似发生在哪里？',['重放段开头缺少更早的局部上下文；它不是从头完整前向的逐值等价结果。','论文通过训练适配减少影响；不能据此断言所有输出完全一致。'],'#fff',INK)
f.save('05-bounded-replay.svg')

f=Figure('06  异步 rollout：长任务跨过训练检查点','示意关注调度关系；不表示固定时长，也不改变原始 prompt 的分组归属。',770)
f.text(40,138,'同一组 GPU 分时执行',23,INK,'650')
for x,w,t,ls,fill,col in [(40,210,'Rollout · θ₀',['样本 A：前半轨迹'],PALE,TEAL),(280,180,'Train 1',['更新到 θ₁'],SAND,GOLD),(490,230,'Rollout · θ₁',['A 恢复 + 新样本'],PALE,TEAL),(750,210,'Train 2',['更新到 θ₂'],SAND,GOLD)]:
 f.box(x,170,w,145,t,ls,fill,col)
f.arrow('250,245 273,245'); f.arrow('460,245 483,245'); f.arrow('720,245 743,245')
f.box(40,365,920,100,'A 的轨迹：生成 → token 级暂停 → 状态保留 → 继续',['KV 与路由状态可以跨检查点保留；沙箱保留任务执行环境。'])
f.box(40,505,440,165,'调度层：按完成数补充新任务',['不必等待整个旧批次的慢样本','完成计数 ≠ 重定义 GRPO 分组'])
f.box(520,505,440,165,'训练层：处理策略滞后与偏差',['限制 off-policy 程度、屏蔽过旧部分','控制长度偏差，避免只留下短任务'],SAND,GOLD)
f.text(40,713,'跨检查点保留 KV 是效率选择；不要把整条恢复轨迹说成严格由当前策略重新采样。',19,MUTED)
f.save('06-async-rollout.svg')
print('Built six static SVG teaching diagrams.')
