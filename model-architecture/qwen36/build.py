"""Editable vector reconstruction and grouped-layer redesign. Run: python3 build.py."""
from pathlib import Path
from html import escape

ROOT = Path(__file__).parent
INK = '#30373d'
BLUE = '#d9eaf5'
GOLD = '#fff2cc'

class Drawing:
    def __init__(self, w, h, title):
        self.w, self.h = w, h
        self.items = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title"><title id="title">{escape(title)}</title>', '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#30373d"/></marker></defs>', '<style>text{font-family:Arial,"Helvetica Neue","PingFang SC",sans-serif;fill:#30373d} .wire{fill:none;stroke:#30373d;stroke-width:1.5;stroke-linejoin:round;stroke-linecap:round}</style>', f'<rect width="{w}" height="{h}" fill="white"/>']
    def text(self,x,y,s,size=15,weight=400,anchor='middle',color=INK):
        self.items.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" style="fill:{color}">{escape(s)}</text>')
    def box(self,x,y,w,h,label='',fill='#fafaf8',stroke='#a5a9ac',r=7,size=15,dash=None):
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
        if label:
            lines=label.split('\n')
            for i,line in enumerate(lines): self.text(x+w/2,y+h/2+5+(i-(len(lines)-1)/2)*(size+4),line,size)
    def path(self,p,arrow=True,dash=None,color=INK):
        self.items.append(f'<path d="{p}" class="wire" style="stroke:{color}"'+(' marker-end="url(#arrow)"' if arrow else '')+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
    def line(self,x1,y1,x2,y2,arrow=True,**kw): self.path(f'M{x1} {y1} L{x2} {y2}',arrow,**kw)
    def circle(self,x,y,s='+',r=11):
        self.items.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{INK}" stroke-width="1.5"/>')
        self.text(x,y+5,s,18)
    def bracket(self,x,y1,y2,label,color=INK):
        self.path(f'M{x-9} {y1} H{x} V{y2} H{x-9}',False,color=color)
        self.text(x+12,(y1+y2)/2+6,label,23,600,'start',color)
    def save(self,name):
        s=''.join(self.items)+'</svg>'
        prefix=name.removesuffix('.svg')
        s=s.replace('id="title"',f'id="{prefix}-title"').replace('aria-labelledby="title"',f'aria-labelledby="{prefix}-title"').replace('id="arrow"',f'id="{prefix}-arrow"').replace('url(#arrow)',f'url(#{prefix}-arrow)')
        (ROOT/name).write_text(s)
        return s

def attention(d,x,y,w=370):
    """Bottom-to-top expanded gated attention, same abstraction as source figure."""
    d.box(x,y,w,290,fill='white',stroke='#929a9f',r=12,dash='2 3')
    d.text(x+18,y+26,'Gated Attention',19,600,'start')
    d.text(x+18,y+47,'全注意力 · QK-Norm · 输出门控',12,400,'start','#68757e')
    cx=x+160
    xs=[x+58,x+136,x+214]
    for a,lab in zip(xs,['Q','K','V']):
        d.box(a-29,y+245,58,27,'Linear',size=12)
        if lab!='V':
            d.line(a,y+245,a,y+223)
            d.box(a-31,y+195,62,28,'RMSNorm',size=10)
            d.line(a,y+195,a,y+164)
        else: d.line(a,y+245,a,y+164)
        d.text(a+10,y+186,lab,12,600,'start')
    d.box(x+26,y+134,228,30,'Scaled dot-product attention',GOLD,size=12)
    d.line(cx,y+134,cx,y+123)
    d.circle(cx,y+111,'×',10)
    d.line(cx,y+101,cx,y+94)
    d.box(cx-27,y+70,54,24,'Linear',size=12)
    d.box(x+284,y+245,60,27,'Linear',size=12)
    d.line(x+314,y+245,x+314,y+201)
    d.circle(x+314,y+189,'σ',10)
    d.path(f'M{x+314} {y+179} V{y+111} H{cx+10}')
    d.text(x+285,y+220,'Sigmoid',11)
    d.text(x+285,y+233,'gate',11)

def delta(d,x,y,w=370):
    d.box(x,y,w,338,fill='white',stroke='#929a9f',r=12,dash='2 3')
    d.text(x+18,y+26,'Gated DeltaNet',19,600,'start')
    d.text(x+18,y+47,'线性注意力 · 递推状态',12,400,'start','#68757e')
    # Separate output path, state update, Q/K/V and scalar gates.
    c=x+166
    d.box(c-26,y+62,52,24,'Linear',size=12)
    d.line(c,y+108,c,y+86)
    d.circle(c,y+120,'×',10)
    d.line(c,y+145,c,y+130)
    d.box(c-35,y+145,70,25,'RMSNorm',size=11)
    d.line(c,y+184,c,y+170)
    d.box(x+25,y+184,250,30,'Gated Delta Rule',GOLD,size=13)
    for i,lab in enumerate(['Q','K','V']):
        xx=x+48+56*i
        d.box(xx-23,y+296,46,25,'Linear',size=10)
        d.line(xx,y+296,xx,y+283)
        d.box(xx-23,y+257,46,26,'Conv', '#f9dad3',size=11)
        d.line(xx,y+257,xx,y+241)
        if lab in ('Q','K'):
            d.box(xx-23,y+221,46,20,'L2 norm',GOLD,size=9)
            d.line(xx,y+221,xx,y+214)
        else: d.line(xx,y+241,xx,y+214)
        d.text(xx,y+250,lab,10,600)
    d.text(x+94,y+335,'Conv 后接 SiLU；Q/K 再做 L2 norm',10)
    for xx,lab in [(x+218,'α'),(x+260,'β')]:
        d.box(xx-17,y+296,34,25,'Proj.',size=9)
        d.line(xx,y+296,xx,y+214)
        d.text(xx+8,y+267,lab,13,600,'start')
    d.box(x+297,y+296,50,25,'Linear',size=11)
    d.line(x+322,y+296,x+322,y+235)
    d.circle(x+322,y+223,'σ',10)
    d.path(f'M{x+322} {y+213} V{y+120} H{c+10}')
    d.text(x+322,y+251,'SiLU gate',10)

def moe(d,x,y,w=370):
    d.box(x,y,w,214,fill='white',stroke='#929a9f',r=12,dash='2 3')
    d.text(x+18,y+27,'Sparse MoE',19,600,'start')
    d.text(x+18,y+48,'256 个路由专家 + 1 个共享专家',12,400,'start','#68757e')
    c=x+184
    d.circle(c,y+76,'+',10)
    d.box(x+26,y+106,158,35,'选中 8 个路由专家',GOLD,size=12)
    d.box(x+212,y+106,135,35,'Shared expert',BLUE,size=12)
    d.path(f'M{x+105} {y+106} V{y+76} H{c-10}')
    d.path(f'M{x+279} {y+106} V{y+76} H{c+10}')
    d.box(x+48,y+165,115,27,'Router · Top-8',size=12)
    d.line(x+105,y+165,x+105,y+141)
    d.path(f'M{c} {y+205} V{y+198} H{x+105} V{y+192}')
    d.path(f'M{c} {y+198} H{x+279} V{y+141}')
    d.text(x+w-12,y+207,'路由输出加权求和 + 共享专家门控输出',10,400,'end','#68757e')

def layer(d,x,y,w,h,name,fill):
    d.box(x,y,w,h,fill=fill,stroke='#a5b3bc',r=12)
    c=x+w*.45
    bx=c-77
    # Every visible layer includes TWO residual connections.
    top=y+26
    d.circle(c,top,'+')
    d.box(bx,y+50,154,30,'MoE',size=14)
    d.line(c,y+50,c,top+11)
    d.box(bx,y+100,154,27,'RMSNorm 2',size=12)
    d.line(c,y+100,c,y+80)
    d.circle(c,y+150,'+')
    d.line(c,y+139,c,y+127)
    d.path(f'M{c} {y+134} H{x+w-28} V{top} H{c+11}')
    d.line(c,y+174,c,y+161)
    d.box(bx-7,y+190,168,36,name,fill='white',stroke='#7d929f',size=14)
    d.line(c,y+190,c,y+174)
    d.box(bx,y+249,154,27,'RMSNorm 1',size=12)
    d.line(c,y+249,c,y+226)
    d.line(c,y+h,c,y+276)
    d.path(f'M{c} {y+h-9} H{x+w-50} V{y+150} H{c+11}')
    d.line(c,top-11,c,y)
    return c

def replica():
    d=Drawing(1120,1100,'Qwen3.6 (35B-A3B) · 可编辑矢量复刻')
    d.text(560,49,'Qwen3.6 (35B-A3B)',32,600,color='#087fa9')
    d.text(560,77,'依据 Sebastian Raschka 的语言模型主干图进行矢量复刻',13,color='#68757e')
    d.box(190,139,350,795,fill='#dddddd',stroke=INK,r=35)
    d.box(245,290,255,480,fill='#91b5d6',stroke=INK,r=30)
    c=365
    d.box(c-68,850,136,31,'Token embedding',size=13)
    d.line(c,850,c,742)
    d.box(c-65,708,130,34,'RMSNorm 1',size=13)
    d.line(c,708,c,655)
    d.box(c-79,592,158,63,'Gated DeltaNet or\nGated Attention',fill='#64717d',stroke=INK,size=13)
    # White label on the dark source-like attention node.
    d.items[-2:]=[s.replace('fill:#30373d','fill:white') for s in d.items[-2:]]
    d.line(c,592,c,554)
    d.circle(c,543)
    d.path(f'M{c} 755 H475 V543 H376')
    d.line(c,532,c,503)
    d.box(c-65,469,130,34,'RMSNorm 2',size=13)
    d.line(c,469,c,431)
    d.box(c-65,397,130,34,'MoE',fill='#e5e5e5',stroke=INK,size=14)
    d.line(c,397,c,348)
    d.circle(c,337)
    d.path(f'M{c} 520 H453 V337 H376')
    d.line(c,326,c,261)
    d.box(c-65,227,130,34,'Final RMSNorm',size=13)
    d.line(c,227,c,201)
    d.box(c-70,170,140,31,'Linear output layer',size=12)
    d.line(c,170,c,108)
    d.text(550,124,'词表大小：约 248k',15,600,'end')
    d.text(177,309,'256 个专家',16,600,'end','#087fa9')
    d.text(177,333,'激活参数 3B',16,600,'end','#087fa9')
    d.line(182,329,302,405,False,dash='2 4')
    d.bracket(228,297,765,'×40','#087fa9')
    d.box(61,600,136,54,'Partial RoPE\n（仅全注意力层）',size=12)
    d.line(197,627,286,627)
    d.text(48,775,'上下文约 262k',16,600,'start','#087fa9')
    d.text(48,799,'扩展后可达 1M',12,400,'start')
    d.line(130,752,145,654,False,dash='2 4')
    d.line(c,965,c,881)
    d.box(c-65,965,130,30,'Tokenized text',size=13)
    d.line(c,1025,c,995)
    d.text(c,1046,'输入文本示例',13)
    d.text(110,960,'Embedding 维度：2,048',14,600,'start')
    d.line(234,937,303,874,False,dash='2 4')
    attention(d,650,142,405)
    delta(d,650,436,405)
    d.line(444,614,650,326,False,dash='2 4')
    d.line(444,637,650,584,False,dash='2 4')
    d.text(670,830,'层类型比例 3:1',20,600,'start')
    for i,s in enumerate(['第 1 层：线性注意力 → MoE','第 2 层：线性注意力 → MoE','第 3 层：线性注意力 → MoE','第 4 层：全注意力 → MoE']): d.text(670,862+i*27,s,15,400,'start')
    d.text(560,1080,'SVG 重绘 · 构图参考 Sebastian Raschka · 非 Qwen 官方图',12,color='#68757e')
    return d.save('qwen36-replica.svg')

def improved():
    d=Drawing(1430,1120,'Qwen3.6-35B-A3B · 3 层 DeltaNet + 1 层 Attention，整组重复 10 次')
    d.text(58,51,'Qwen3.6–35B-A3B',32,600,'start','#16799a')
    d.text(58,81,'语言模型主干 · 明确层排列与重复次数',16,400,'start','#657782')
    d.text(1372,48,'总参数 35B / 激活参数 3B',16,600,'end')
    d.text(1372,76,'40 层 = (3 + 1) × 10',18,600,'end','#16799a')
    # Left: actual stack, bottom to top, with nested repetition like the user reference.
    d.box(126,173,423,753,fill='#fbfcfc',stroke='#939fa7',r=16,dash='5 5')
    d.text(148,198,'每组 4 层',14,600,'start','#657782')
    d.bracket(568,178,920,'×10','#16799a')
    d.box(144,548,385,360,fill='none',stroke='#a59260',r=13,dash='2 4')
    d.bracket(480,572,887,'×3','#90701e')
    d.text(166,570,'连续 3 层线性注意力',13,600,'start','#90701e')
    c=layer(d,164,580,296,306,'Gated DeltaNet',GOLD)
    d.box(144,214,385,319,fill='none',stroke='#7194ac',r=13,dash='2 4')
    d.text(478,241,'×1',21,600,'start','#397999')
    layer(d,164,222,296,306,'Gated Attention',BLUE)
    d.line(c,580,c,528)
    d.text(478,274,'层编号',12,400,'start','#397999')
    d.text(478,294,'4g + 4',12,600,'start','#397999')
    d.line(c,222,c,168)
    d.box(c-80,136,160,32,'Final RMSNorm',size=13)
    d.line(c,136,c,125)
    d.box(c-80,100,160,25,'LM head → logits',size=13)
    d.box(c-80,951,160,32,'Token embedding',size=13)
    d.line(c,951,c,886)
    d.line(c,1010,c,983)
    d.text(c,1033,'Token IDs · 隐藏维度 2,048',14)
    d.text(610,954,'g = 0 … 9',13,400,'end','#657782')
    d.text(610,979,'30 层 DeltaNet + 10 层 Attention',13,600,'end')
    # Detail panels, same colors as corresponding block. No arrow through main graph.
    attention(d,690,127,360)
    delta(d,690,440,360)
    moe(d,690,800,360)
    # Dedicated explanatory rail, spare white space and no crossing leader lines.
    d.text(1100,150,'01  全注意力层',13,600,'start','#397999')
    d.text(1100,180,'第 4、8、12 … 40 层',16,600,'start')
    d.text(1100,209,'Q：16 个头 · KV：2 个头',14,400,'start')
    d.text(1100,236,'每个头的维度：256',14,400,'start')
    d.text(1100,263,'Q/K 使用 Partial RoPE',14,400,'start')
    d.text(1100,290,'仅作用于 256 维中的 64 维',14,400,'start')
    d.text(1100,448,'02  线性注意力层',13,600,'start','#90701e')
    d.text(1100,478,'第 4g+1、4g+2、4g+3 层',16,600,'start')
    d.text(1100,507,'Q/K：16 个头 · V：32 个头',14,400,'start')
    d.text(1100,534,'每个头的维度：128',14,400,'start')
    d.text(1100,561,'Depthwise Conv1D · k = 4',14,400,'start')
    d.text(1100,588,'DeltaNet 层不使用 RoPE',14,400,'start')
    d.text(1100,816,'03  每层共有的结构',13,600,'start','#657782')
    d.text(1100,846,'MoE + 两条残差路径',16,600,'start')
    d.text(1100,875,'激活 8 个路由专家 + 1 个共享专家',14,400,'start')
    d.text(1100,902,'专家 FFN 使用 SwiGLU',14,400,'start')
    d.text(1100,929,'专家中间维度：512',14,400,'start')
    d.line(58,1054,1372,1054,False,color='#ced6dc')
    d.text(58,1080,'范围：文本 Decoder 主干。视觉编码器与辅助 MTP 模块未展开。',12,400,'start','#657782')
    d.text(58,1100,'依据：Qwen/Qwen3.6-35B-A3B 配置与模型卡；层编号从 1 开始；构图参考 Raschka 与用户提供的重复框示意。',11,400,'start','#657782')
    return d.save('qwen36-improved.svg')

if __name__=='__main__':
    a,b=replica(),improved()
    template=(ROOT/'template.html').read_text()
    (ROOT/'index.html').write_text(template.replace('<!-- REPLICA -->',a).replace('<!-- IMPROVED -->',b))
    print('Generated index.html and two editable SVGs')
