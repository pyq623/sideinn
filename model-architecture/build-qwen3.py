"""Build the two Qwen3 architecture pages. Standard-library only; no network at build time."""
from pathlib import Path
from html import escape
import json
import drawing
from drawing import Drawing
ROOT=Path(__file__).parent
BLUE='#dcebf5'; GOLD='#fff1c9'; GREEN='#e3eee7'; MUTED='#687c88'
MODELS=[dict(slug='qwen3-30b-a3b',name='Qwen3-30B-A3B',moe=True,number='02'),dict(slug='qwen3-8b',name='Qwen3-8B',moe=False,number='03')]

def note(d,x,y,lines,size=14):
    for i,s in enumerate(lines):d.text(x,y+i*(size+10),s,size,400,'start',MUTED)

def swiglu(d,x,y,w,hidden,inter,title='SwiGLU FFN'):
    d.box(x,y,w,292,fill='#f8faf8',stroke='#adbab6',dash='3 4',r=12)
    d.text(x+20,y+28,title,18,600,'start')
    c=x+w/2; left=c-75; right=c+75
    d.box(c-70,y+48,140,30,'Down projection',fill=GREEN,size=13)
    d.line(c,y+103,c,y+78); d.circle(c,y+115,'×')
    d.path(f'M{left} {y+152} V{y+115} H{c-11}')
    d.path(f'M{right} {y+205} V{y+115} H{c+11}')
    d.box(left-59,y+152,118,30,'SiLU',fill=GOLD,size=13)
    d.line(left,y+205,left,y+182)
    d.box(left-59,y+205,118,30,'Gate projection',size=12)
    d.box(right-59,y+205,118,30,'Up projection',size=12)
    d.path(f'M{c} {y+269} V{y+251} H{left} V{y+235}')
    d.path(f'M{c} {y+251} H{right} V{y+235}')
    d.text(c,y+286,f'{hidden:,} → {inter:,} → {hidden:,}',12)

def replica(m,c):
    d=Drawing(1200,1200,m['name']+' · 原布局复刻 · 可编辑矢量图')
    d.text(55,57,m['name'],36,650,'start','#2382a9')
    d.text(55,85,'原布局复刻 · 自下而上的主干 + 右侧模块详情',15,400,'start',MUTED)
    d.box(150,130,465,930,fill='#f0f1ef',r=30)
    d.box(220,295,315,555,fill=BLUE,stroke='#7d9aac',r=28)
    center=365
    # All connections are outside the node rectangles.
    d.box(260,1085,210,34,'Tokenized text');d.line(center,1085,center,1020)
    d.box(240,980,250,40,'Token embedding');d.line(center,980,center,815)
    for y,label in [(775,'RMSNorm 1'),(660,'Grouped-query attention'),(535,'RMSNorm 2'),(445,'MoE' if m['moe'] else 'Feed forward')]:
        d.box(255,y,220,40,label,fill=GOLD if y==445 else 'white',size=14)
    d.line(center,775,center,700);d.line(center,660,center,621);d.circle(center,610)
    d.line(center,599,center,575);d.line(center,535,center,485);d.line(center,445,center,366);d.circle(center,355)
    d.path(f'M{center} 830 H505 V610 H376');d.path(f'M{center} 588 H505 V355 H376')
    d.box(270,235,190,35,'Final RMSNorm');d.line(center,344,center,270)
    d.box(255,165,220,35,'Linear output layer');d.line(center,235,center,200);d.line(center,165,center,115)
    d.bracket(195,295,850,'',color='#7b9aab');d.text(182,880,f'×{c["num_hidden_layers"]}',26,600,'middle','#2382a9')
    d.box(42,615,145,34,'QK-Norm');d.path('M187 632 L250 670')
    d.box(42,690,145,34,'RoPE');d.line(187,707,250,690)
    d.text(490,179,'151,936',13,600,'start','#2382a9')
    d.text(455,1046,f'd = {c["hidden_size"]:,}',14,400,'start')
    swiglu(d,730,160,410,c['hidden_size'],c.get('moe_intermediate_size',c['intermediate_size']),'Feed forward (SwiGLU)')
    if m['moe']:
        d.path('M475 465 L700 610',False,dash='4 5',color='#4195b8')
        d.box(700,560,450,385,fill='white',stroke='#4195b8',dash='4 5',r=25)
        d.text(725,594,'MoE layer',22,600,'start')
        d.circle(925,650,'+');d.line(925,639,925,615)
        for xx,lab in [(790,'Expert 1'),(1060,'Expert 128')]:
            d.box(xx-64,715,128,38,lab,fill=GOLD,size=14)
            d.line(xx,715,915 if xx<925 else 935,658)
            d.line(925,838,xx,753)
        d.text(925,740,'···',28,600)
        d.box(860,838,130,38,'Router',fill=BLUE);d.line(925,910,925,876)
        note(d,730,985,['128 个专家 · 每个 token 选 8 个','总参数 30.5B · 激活参数 3.3B','每个专家的中间维度：768'])
    else:
        d.path('M475 465 L730 360',False,dash='4 5')
        note(d,730,560,['每层均使用 Dense FFN','32 个 Q 头 · 8 个 KV 头','每个头的维度：128','隐藏维度：4,096','FFN 中间维度：12,288'],18)
        d.box(730,790,410,136,fill='#f7f9fa',stroke='#d3dde3',r=12)
        note(d,750,825,['36 层均采用全注意力','每层均包含 QK-Norm 与 RoPE','不使用专家路由'],16)
    d.line(55,1145,1145,1145,False,color='#d6dfe3')
    note(d,55,1171,['构图参考：Sebastian Raschka · LLM Architecture Gallery。可编辑 SVG 重绘，非像素级复制。'],12)
    return d.save(m['slug']+'-replica.svg')

def attention_detail(d,m,c):
    x,y,w,h=590,145,790,420
    d.box(x,y,w,h,fill='#f9fbfc',stroke='#a8bfcd',r=14)
    d.text(x+24,y+32,'01  GROUPED-QUERY ATTENTION',19,600,'start','#2f6f91')
    # Projection → per-head RMSNorm → RoPE → attention → output projection.
    q,k,v=x+88,x+222,x+356; center=x+222
    d.box(center-125,y+62,250,32,f'Output projection  4096 → {c["hidden_size"]}',fill=BLUE,size=13)
    d.line(center,y+120,center,y+94)
    d.box(x+24,y+120,400,38,'Causal attention · 拼接 32 个头的输出',fill=BLUE,size=14)
    for xx,label,dim in [(q,'Q',4096),(k,'K',c['num_key_value_heads']*128),(v,'V',c['num_key_value_heads']*128)]:
        d.box(xx-57,y+302,114,32,f'{label} projection\n→ {dim}',size=11)
        d.line(xx,y+358,xx,y+334)
        if label!='V':
            d.box(xx-57,y+242,114,32,'逐头 RMSNorm',size=11)
            d.line(xx,y+302,xx,y+274)
            d.box(xx-57,y+185,114,30,'RoPE',fill=GOLD,size=13)
            d.line(xx,y+242,xx,y+215);d.line(xx,y+185,xx,y+158)
        else:d.line(xx,y+302,xx,y+158)
    d.path(f'M{q} {y+358} H{v}',False);d.line(center,y+379,center,y+358)
    d.text(center,y+402,'输入：RMSNorm 1 的输出',12,400,'middle',MUTED)
    d.line(x+452,y+70,x+452,y+388,False,color='#d4e0e7')
    note(d,x+477,y+95,[f'Q：32 个头 × 128 维',f'K/V: {c["num_key_value_heads"]} 个头 × 128 维',f'{32//c["num_key_value_heads"]} 个 Q 头共用一组 KV','Q/K 在每个头的 128 维上归一化','RoPE 作用于 Q 和 K','V 不经过 QK-Norm 或 RoPE','每层都使用完整因果注意力','注意力输出不额外门控'],14)

def moe_detail(d,c):
    x,y=590,595
    d.box(x,y,790,370,fill='#fffcf3',stroke='#cdbd8c',r=14)
    d.text(x+24,y+32,'02  Sparse MoE · 128 个专家，选 8 个',19,600,'start','#8c6b21')
    # Routing is a selector; normalized token features also go directly to the experts.
    d.box(x+40,y+72,320,34,'对选中专家的输出加权求和',fill=GOLD,size=12)
    for xx,lab in [(x+108,'Expert i₁'),(x+288,'Expert i₈')]:
        d.box(xx-60,y+146,120,48,lab+'\nSwiGLU',fill='white',size=13)
        d.line(xx,y+146,xx,y+106)
        d.line(xx,y+256,xx,y+194)
    d.text(x+198,y+172,'···',25,600)
    d.path(f'M{x+108} {y+256} H{x+288}',False)
    d.box(x+40,y+293,320,32,'输入：RMSNorm 2 的输出',size=13)
    d.line(x+108,y+293,x+108,y+256)
    d.box(x+425,y+243,320,70,'Router: Linear → softmax\nTop-8 → 对选中权重重新归一化',fill=GOLD,size=13)
    d.path(f'M{x+360} {y+309} H{x+395} V{y+278} H{x+425}')
    d.path(f'M{x+425} {y+264} H{x+385} V{y+176} H{x+348}',dash='4 4',color='#997b30')
    d.path(f'M{x+585} {y+243} H{x+770} V{y+89} H{x+360}',dash='4 4',color='#997b30')
    note(d,x+425,y+130,['每个专家：2,048 → 768 → 2,048','48 层全部使用 Sparse MoE','无共享专家，也无稠密前置层','虚线：专家选择与路由权重'],14)
    d.text(x+40,y+351,'实线传递特征；Router 为每个 token 从 128 个专家 FFN 中选出 8 个。',12,400,'start',MUTED)

def improved(m,c):
    n=c['num_hidden_layers'];hidden=c['hidden_size']
    d=Drawing(1440,1240,m['name']+' · 完整层、层编号、注意力与前馈网络')
    d.text(55,57,m['name'],34,650,'start','#286d8c')
    d.text(55,87,'完整层与重复范围 · 数据流自下而上 · 层编号从 1 开始',16,400,'start',MUTED)
    d.text(1380,52,f'{n} 层  /  隐藏维度 {hidden:,}',14,600,'end',MUTED)
    # Complete decoder layer; output components are outside the repeated boundary.
    d.box(70,260,400,610,fill='#f8fafb',stroke='#90a8b7',dash='5 5',r=16)
    d.text(93,286,'一个完整的',13,600,'start',MUTED)
    d.text(93,307,'Decoder 层',13,600,'start',MUTED)
    d.bracket(495,260,870,f'×{n}',color='#2f6f91')
    center=265
    d.box(152,145,226,36,f'LM head → {c["vocab_size"]:,} logits',size=13)
    d.box(170,208,190,32,'Final RMSNorm',size=14)
    d.line(center,208,center,181)
    d.line(center,322,center,240)
    d.circle(center,333)
    d.box(150,380,230,50,'Sparse MoE · top-8 / 128' if m['moe'] else 'Dense SwiGLU FFN',fill=GOLD,size=14)
    d.line(center,380,center,344)
    d.box(177,477,176,34,'RMSNorm 2',size=14);d.line(center,477,center,430)
    d.circle(center,564);d.line(center,553,center,511)
    d.path(f'M{center} 538 H423 V333 H276')
    d.text(414,449,'残差连接',11,400,'end',MUTED)
    d.box(150,622,230,54,'Grouped-query attention\nQK-Norm + RoPE',fill=BLUE,size=14)
    d.line(center,622,center,575)
    d.box(177,737,176,34,'RMSNorm 1',size=14);d.line(center,737,center,676)
    d.line(center,920,center,771)
    d.path(f'M{center} 827 H423 V564 H276');d.text(414,714,'残差连接',11,400,'end',MUTED)
    d.text(95,851,'2 次前置归一化',12,400,'start',MUTED)
    d.text(305,851,'2 次残差相加',12,400,'start',MUTED)
    d.box(146,920,238,38,f'Token embedding · d = {hidden:,}',size=13)
    d.line(center,995,center,958);d.text(center,1015,'Token IDs',14)
    d.text(70,1050,f'全部 {n} 层均采用此结构。',15,600,'start','#2f6f91')
    d.text(70,1075,'各层分别学习参数，不共享权重。',13,400,'start',MUTED)
    attention_detail(d,m,c)
    if m['moe']:moe_detail(d,c)
    else:
        swiglu(d,590,595,420,hidden,c['intermediate_size'],'02  DENSE SwiGLU FFN')
        note(d,1040,635,['Gate projection：4,096 → 12,288','Up projection：4,096 → 12,288','Down projection：12,288 → 4,096','Gate projection 的输出先经过 SiLU，','再与 Up projection 的输出逐元素相乘，','随后交给 Down projection。','同层所有 token 使用同一个 FFN；','不使用 Router 或稀疏专家。'],14)
        d.box(590,909,790,56,'36 层各有一个 Dense FFN · 各层参数独立',fill=GREEN,size=15)
    d.text(590,1003,'层序列（每格 = 上方左侧的一个完整层）',15,600,'start',MUTED)
    cols=24 if m['moe'] else 18; cell=30 if m['moe'] else 41
    for i in range(n):
        xx=590+(i%cols)*(cell+3); yy=1023+(i//cols)*37
        d.box(xx,yy,cell,29,f'{i+1:02}',fill=BLUE,size=11,r=4)
    d.text(590,1122,'所有层类型相同；重复范围包含 Attention、FFN 和两条残差路径。',12,400,'start',MUTED)
    d.line(55,1160,1385,1160,False,color='#d5dfe4')
    note(d,55,1185,['依据：Qwen 官方配置与 Transformers v4.51.3 实现。图中为纯文本 Decoder 主干。',
                       '上下文：模型卡标注原生 32,768 token，YaRN 扩展至 131,072；配置中的 max_position_embeddings = 40,960。'],12)
    return d.save(m['slug']+'-improved.svg')

def build(m):
    out=ROOT/m['slug'];c=json.loads((out/'config.snapshot.json').read_text());drawing.ROOT=out
    a,b=replica(m,c),improved(m,c)
    ffn='稀疏 MoE' if m['moe'] else '稠密 SwiGLU FFN'
    desc=f'{c["num_hidden_layers"]} 个完整 Decoder 层，每层包含 GQA、{ffn}、两次归一化和两条残差连接。'
    meta='128 个专家 · 每 token 选 8 个 · 无共享专家' if m['moe'] else '32 个 Q 头 / 8 个 KV 头 · FFN 中间维度 12,288'
    vals={'MODEL':m['name'],'SLUG':m['slug'],'NUMBER':m['number'],'DESCRIPTION':desc,'META':meta,'LAYERS':str(c['num_hidden_layers']),'REPLICA':a,'IMPROVED':b,'MODULE':'qwen3_moe' if m['moe'] else 'qwen3'}
    text=(ROOT/'qwen3-template.html').read_text()
    for k,v in vals.items():text=text.replace('{{'+k+'}}',v)
    (out/'index.html').write_text(text)
    source={'model':m['name'],'checked':'2026-11-02','config':f'https://huggingface.co/Qwen/{m["name"]}/blob/main/config.json','reference':f'https://sebastianraschka.com/llm-architecture-gallery/images/architectures/{m["slug"]}.webp','implementation':f'https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/{vals["MODULE"]}/modeling_{vals["MODULE"]}.py','scope':'Original Qwen3 checkpoint (April 2025), not 2507, Coder, Next, or Qwen3.6.'}
    (out/'sources.json').write_text(json.dumps(source,ensure_ascii=False,indent=2)+'\n')
    print('Built',m['name'])
if __name__=='__main__':
    for m in MODELS:build(m)
