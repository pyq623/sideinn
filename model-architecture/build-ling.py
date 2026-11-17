"""Build editable Flash diagrams from the pinned Bailing v2.5 configuration."""
from pathlib import Path
import json
import drawing
from drawing import Drawing

ROOT = Path(__file__).parent
OUT = ROOT / 'ling-2.6-flash'
C = json.loads((OUT / 'config.snapshot.json').read_text())
S = json.loads((OUT / 'sources.json').read_text())
drawing.ROOT = OUT
BLUE = '#dfedf6'
GREEN = '#e3eee7'
GOLD = '#fff0cc'
MUTED = '#647985'
SLUG = 'ling-2.6-flash'

def notes(d,x,y,lines,size=14):
    for i,t in enumerate(lines): d.text(x,y+i*(size+11),t,size,400,'start',MUTED)

def heading(d,subtitle,w):
    d.text(48,54,'Ling-2.6-flash',34,650,'start','#286d8c')
    d.text(48,86,subtitle,16,400,'start',MUTED)
    d.text(w-48,47,'Bailing v2.5 架构',16,600,'end',MUTED)
    d.text(w-48,75,'约 104B 总参数 / 7.4B 激活参数',14,400,'end',MUTED)

def layer(d,x,y,w,attn,ffn):
    """A complete pre-norm layer, bottom-up; bounds y..y+360."""
    d.box(x,y,w,360,fill=BLUE if attn=='MLA' else GREEN,stroke='#91aab6',r=14)
    cx=x+w/2; bw=w-100
    for yy,label,fill in [(y+290,'RMSNorm', 'white'),(y+230,attn,'white'),(y+135,'RMSNorm','white'),(y+75,ffn,GOLD)]:
        d.box(cx-bw/2,yy,bw,34,label,fill=fill,size=14)
    d.line(cx,y+360,cx,y+324); d.line(cx,y+290,cx,y+264)
    d.line(cx,y+230,cx,y+201); d.circle(cx,y+190)
    d.line(cx,y+179,cx,y+169); d.line(cx,y+135,cx,y+109)
    d.line(cx,y+75,cx,y+46); d.circle(cx,y+35)
    d.line(cx,y+24,cx,y)
    d.path(f'M{cx} {y+343} H{x+w-20} V{y+190} H{cx+11}')
    d.path(f'M{cx} {y+175} H{x+w-20} V{y+35} H{cx+11}')

def moe(d,x,y,w=700):
    d.box(x,y,w,400,fill='#fffcf5',stroke='#cbbb92',r=14)
    d.text(x+24,y+32,'MoE · 256 个路由专家 + 1 个共享专家',20,600,'start','#866722')
    cx=x+w/2
    d.circle(cx,y+76); d.line(cx,y+65,cx,y+48)
    d.box(x+32,y+142,170,56,'Shared expert\nSwiGLU',fill=GREEN,size=14)
    d.box(x+270,y+142,w-302,56,'Top-8 routed experts · SwiGLU\n对选中专家的输出加权求和',fill=GOLD,size=14)
    d.path(f'M{x+117} {y+142} V{y+76} H{cx-11}')
    d.path(f'M{x+w-170} {y+142} V{y+76} H{cx+11}')
    d.box(x+32,y+277,220,34,'输入：RMSNorm 的输出',size=13)
    d.line(x+117,y+277,x+117,y+198)
    d.path(f'M{x+252} {y+294} H{x+278} V{y+225} H{x+w-170} V{y+198}')
    d.box(x+330,y+265,w-362,62,'Router → sigmoid\n分组选择 → Top-8',fill=BLUE,size=14)
    d.line(x+252,y+294,x+330,y+294)
    d.path(f'M{x+w-125} {y+265} V{y+210} H{x+w-140} V{y+198}',dash='4 4')
    notes(d,x+24,y+350,['实线传递特征；虚线传递选择与权重。共享专家处理每个 token。',
        'Bias 只参与选择；原始 sigmoid 分数归一化后乘 2.5 作为路由权重。'],13)

def lightning(d,x,y,w=700):
    d.box(x,y,w,585,fill='#f5f9fb',stroke='#a6bdca',r=14)
    d.text(x+24,y+33,'Lightning Attention · 内部数据流',20,600,'start','#286d8c')
    cx=x+280; gate=x+w-108
    for yy,label in [(y+68,'Output projection'),(y+175,'Group RMSNorm'),(y+242,'Lightning Attention kernel'),(y+325,'Partial RoPE · Q / K'),(y+388,'QK-Norm · Q / K'),(y+454,'QKV projection')]:
        d.box(cx-128,yy,256,34,label,fill=BLUE if yy==y+242 else 'white',size=14)
    d.line(cx,y+175,cx,y+148);d.circle(cx,y+137,'×');d.line(cx,y+126,cx,y+102)
    d.line(cx,y+242,cx,y+209)
    d.line(cx,y+325,cx,y+276);d.line(cx,y+388,cx,y+359);d.line(cx,y+454,cx,y+422)
    d.path(f'M{cx+128} {y+471} H{x+438} V{y+259} H{cx+128}')
    d.text(x+448,y+365,'V',14,600,'start')
    d.box(gate-66,y+454,132,34,'Gate projection',size=13)
    d.box(gate-66,y+325,132,34,'sigmoid',size=14)
    d.line(gate,y+454,gate,y+359)
    d.path(f'M{gate} {y+325} V{y+137} H{cx+11}')
    d.path(f'M{cx} {y+535} V{y+512} H{gate} V{y+488}')
    d.line(cx,y+512,cx,y+488)
    d.text(cx,y+557,'输入：层内第一个 RMSNorm 的输出',13,400,'middle',MUTED)
    notes(d,x+22,y+83,['32 个头','每头 128 维'],12)

def overview():
    d=Drawing(1380,1350,'Ling-2.6-flash · 参考布局重绘；按 Flash 配置适配')
    heading(d,'参考官方 1T 图的构图 · 按 Flash 配置适配 · 数据流自下而上',1380)
    d.box(65,140,505,1080,fill='#f8fafb',stroke='#b3c2ca',r=22)
    cx=310
    d.box(200,175,220,36,'LM Head · 157,184',size=14)
    d.box(200,235,220,34,'Final RMSNorm',size=14);d.line(cx,235,cx,211)
    d.box(100,305,420,800,fill='none',stroke='#748f9e',dash='4 4',r=18)
    layer(d,150,330,320,'MLA','MoE');d.line(cx,330,cx,269)
    layer(d,150,715,320,'Lightning Attention','FFN：通常为 MoE');d.line(cx,715,cx,690)
    d.bracket(487,330,690,'×1');d.bracket(487,715,1075,'×7')
    d.text(123,1092,'整组 ×4',18,600,'start','#286d8c')
    d.box(200,1145,220,34,'Token embedding',size=14);d.line(cx,1145,cx,1075)
    d.box(200,1250,220,34,'Tokenized text',size=14);d.line(cx,1250,cx,1179)
    moe(d,625,140)
    lightning(d,625,570)
    notes(d,625,1195,['32 层 = 28 层 Lightning Attention + 4 层 MLA',
        '第 1 层的 FFN 使用 Dense SwiGLU；其余 31 层使用 MoE。',
        '图形构图参考官方 1T 图，层数与参数均按 Flash 配置改写。',
        '此图为资料重绘，并非官方发布的 Flash 原图。'],15)
    notes(d,48,1320,['仅绘制 32 层 Decoder 主干；不展开辅助 MTP。×N 表示结构重复，各层参数独立。'],13)
    return d.save(SLUG+'-replica.svg')

def block(d,x,y,label,sub,fill):
    d.box(x,y,320,82,fill=fill,stroke='#9fb3bd',r=9)
    d.text(x+160,y+32,label,17,600)
    d.text(x+160,y+59,sub,14)

def improved():
    d=Drawing(1440,1450,'Ling-2.6-flash · 32 层完整分组与 Dense FFN 例外')
    heading(d,'完整层展开 · 第一组单独展示 · 数据流自下而上，层编号从 1 开始',1440)
    cx=285
    d.box(175,130,220,36,'LM Head · 157,184',size=14)
    d.line(cx,130,cx,112)
    d.box(175,195,220,34,'Final RMSNorm',size=14);d.line(cx,195,cx,166)
    # Three homogeneous groups, each with seven complete linear layers and one MLA layer.
    d.box(55,275,470,345,fill='#f8fafb',stroke='#91aab6',dash='5 5',r=14)
    d.text(78,307,'第 2–4 组 · 第 9–32 层',18,600,'start')
    block(d,125,340,'MLA + MoE','完整层 ×1 · 第 16 / 24 / 32 层',BLUE)
    block(d,125,480,'Lightning Attention + MoE','完整层 ×7 · 每组前 7 层',GREEN)
    d.line(cx,480,cx,422);d.line(cx,340,cx,229)
    d.bracket(545,275,620,'×3')
    # Exceptional first group, containing exactly 1 dense + 6 MoE linear + 1 MLA MoE.
    d.box(55,660,470,450,fill='#fbfcfc',stroke='#91aab6',dash='5 5',r=14)
    d.text(78,692,'第 1 组 · 第 1–8 层',18,600,'start')
    block(d,125,720,'MLA + MoE','完整层 ×1 · 第 8 层',BLUE)
    block(d,125,850,'Lightning Attention + MoE','完整层 ×6 · 第 2–7 层',GREEN)
    block(d,125,980,'Lightning Attention + Dense FFN','完整层 ×1 · 仅第 1 层',GOLD)
    d.line(cx,850,cx,802);d.line(cx,980,cx,932);d.line(cx,720,cx,561)
    d.bracket(545,660,1110,'×1')
    d.box(175,1155,220,36,'Token embedding',size=14);d.line(cx,1155,cx,1062)
    d.text(cx,1225,'Tokenized text → 隐藏维度 4,096',14)
    d.line(cx,1205,cx,1191)
    # The left grouped modules all refer to this exact complete residual layer.
    d.box(660,130,730,475,fill='#fafcfd',stroke='#aabec9',r=14)
    d.text(684,163,'01  一个完整 Decoder 层',20,600,'start','#286d8c')
    layer(d,690,205,325,'Attention','Dense FFN / MoE')
    notes(d,1045,224,['左侧每个「完整层」均包含：',
        'RMSNorm → Attention → 残差相加',
        'RMSNorm → FFN → 残差相加','',
        'Attention 按层号选择：',
        'Lightning Attention 或 MLA','',
        'FFN 按层号选择：',
        '第 1 层：Dense SwiGLU',
        '第 2–32 层：MoE'],14)
    d.text(684,586,'Embedding、Final RMSNorm、LM Head 均在重复边界之外。',13,400,'start',MUTED)
    moe(d,660,635,730)
    d.box(660,1065,730,180,fill='#f5f9fb',stroke='#aabec9',r=14)
    d.text(684,1097,'02  两种 Attention 与专家维度',19,600,'start','#286d8c')
    notes(d,684,1126,['Lightning Attention：32 个头 × 128 维；QK-Norm、Partial RoPE、输出门控。',
        'MLA：Q 压缩维度 1,536，KV 压缩维度 512；32 个注意力头。',
        '每个 routed / shared expert 均为 SwiGLU：4,096 → 1,024 → 4,096。',
        'MoE 的 8 个路由专家之外，另有 1 个共享专家始终参与计算。',
        '首层 Dense SwiGLU：4,096 → 9,216 → 4,096。'],14)
    d.text(48,1286,'全部 32 层 · 蓝色为 MLA，绿色为 Lightning Attention；金色为首层 Dense FFN',15,600,'start')
    for i in range(32):
        fill=GOLD if i==0 else BLUE if (i+1)%8==0 else GREEN
        d.box(48+i*42,1305,36,34,f'{i+1:02}',fill=fill,size=13,r=4)
    notes(d,48,1381,['32 = (1 + 6 + 1) + (7 + 1) ×3；共 28 层 Lightning Attention、4 层 MLA、1 层 Dense FFN、31 层 MoE。',
        '依据官方固定版本配置与实现重绘。×N 代表重复结构，各层参数独立；仅展示 Decoder 主干，不展开辅助 MTP。'],14)
    return d.save(SLUG+'-improved.svg')

def build():
    assert C['num_hidden_layers']==32 and C['layer_group_size']==8 and C['first_k_dense_replace']==1
    assert C['num_experts']==256 and C['num_experts_per_tok']==8 and C['num_shared_experts']==1
    a,b=overview(),improved()
    t=(OUT/'template.html').read_text()
    for k,v in {'REPLICA':a,'IMPROVED':b,'CONFIG':S['config'],'IMPLEMENTATION':S['implementation'],'REVISION':S['revision'][:12]}.items():
        t=t.replace('{{'+k+'}}',v)
    (OUT/'index.html').write_text(t)
    print('Built Ling-2.6-flash: 2 SVG diagrams and HTML viewer')

if __name__=='__main__': build()
