"""Editable redraw of Qwen's architecture preview (Figure 1), not Qwen4-Max."""
from pathlib import Path
import json
import drawing
from drawing import Drawing

ROOT = Path(__file__).parent
OUT = ROOT / 'qwen4-preview'
SLUG = 'qwen4-preview'
C = json.loads((OUT / 'config.snapshot.json').read_text())['text_config']
S = json.loads((OUT / 'sources.json').read_text())
drawing.ROOT = OUT
GREEN, BLUE, PURPLE = '#edf3e7', '#e8eef6', '#eee3f2'
READ, WRITE, MOE = '#ffddb0', '#d9e9fa', '#f3dadd'
MUTED, TEAL = '#667783', '#246b79'


def notes(d, x, y, lines, size=15):
    for i, line in enumerate(lines):
        d.text(x, y + i * (size + 12), line, size, anchor='start', color=MUTED)


def heading(d, subtitle, width):
    d.text(44, 53, 'Qwen4 架构预览', 32, 650, 'start', TEAL)
    d.text(44, 86, subtitle, 15, anchor='start', color=MUTED)
    d.text(width-44, 49, 'Qwen3.8-Flash-Next', 21, 600, 'end')
    d.text(width-44, 80, '语言模型主干 · 不代表 Qwen4-Max 配置', 14, anchor='end', color=MUTED)


def residual(d, x, y):
    d.box(x-64, y-16, 128, 32, fill='white', r=3)
    for i in range(4):
        d.circle(x-39+i*26, y, '', 6)


def detail(d, x, y, kind, width=710):
    """One complete layer; GR includes pre-normalization (report §2.2)."""
    d.box(x, y, width, 390, fill=BLUE if kind=='QSA Layer' else GREEN, stroke='#91a3ac', r=22)
    d.text(x+22, y+29, kind, 20, 600, 'start')
    rx, gx, hx, op = x+115, x+345, x+480, x+590
    for yy in (y+60, y+220, y+345):
        residual(d, rx, yy)
    for base, label in ((y+345, 'Qwen Sparse Attn' if kind=='QSA Layer' else 'Gated DeltaNet'), (y+220, 'MoE')):
        out = y+220 if base==y+345 else y+60
        read_y = base if base==y+345 else y+160
        # The unchanged residual enters GR Write; the processed hidden state joins it.
        d.line(rx, base-16, rx, out+16, arrow=False)
        d.path(f'M{rx+64 if base==y+345 else rx} {read_y} H{gx-57}')
        d.box(gx-57, read_y-16, 114, 32, 'GR Read', READ, size=16, r=4)
        d.line(gx+57, read_y, hx-18, read_y)
        d.box(hx-18, read_y-16, 36, 32, '○', 'white', r=3)
        d.path(f'M{hx+18} {read_y} H{op} V{(read_y+out)/2+18}')
        d.box(op-91, (read_y+out)/2-18, 182, 36, label, MOE if label=='MoE' else '#f5f8ed', size=15, r=4)
        d.path(f'M{op} {(read_y+out)/2-18} V{out} H{hx+18}')
        d.box(hx-18, out-16, 36, 32, '○', 'white', r=3)
        d.line(hx-18, out, gx+57, out)
        d.box(gx-57, out-16, 114, 32, 'GR Write', WRITE, size=16, r=4)
        d.line(gx-57, out, rx+64, out)
        d.path(f'M{rx} {read_y-34} H{gx} V{out+16}')
    d.text(rx, y+379, 'Expanded Residual', 14)
    d.text(hx+60, y+379, 'Hidden States', 14)


def replica():
    d = Drawing(1380,1420,'Qwen4 架构预览 · 官方图布局复刻 · Qwen3.8-Flash-Next')
    heading(d,'01 官方图布局复刻 · 保留双栏构图、模块名称与连接方向 · 数据流自下而上',1380)
    cx=190
    d.box(95,134,190,44,'MTP Modules','#fbefcf',size=17)
    d.box(705,134,225,44,'Prediction Head','#fbefcf',size=17)
    d.box(615,220,114,32,'GR Read',READ,size=16)
    d.box(800,220,36,32,'○','white')
    d.path('M190 274 V236 H615'); d.line(729,236,800,236); d.line(818,220,818,178)
    d.line(190,236,190,178)
    d.box(100,274,180,34,'…','white')
    for n in range(12):
        yy=345+n*60
        qsa=n%4==0
        d.box(100,yy,180,34,'QSA Layer' if qsa else 'GDN Layer',BLUE if qsa else GREEN,size=16,r=3)
        d.line(cx,yy,cx,yy-26 if n else 308)
    d.box(84,572,212,248,fill='none',stroke='#8ba5b4',dash='6 5',r=0)
    d.path('M296 572 L450 286',False,dash='5 5'); d.path('M296 820 L450 1150',False,dash='5 5')
    d.box(450,286,870,864,fill='none',stroke='#8ba5b4',dash='6 5',r=0)
    d.text(1305,277,'Hybrid Block × L/4',16,600,'end')
    detail(d,590,308,'QSA Layer'); detail(d,590,738,'GDN Layer')
    d.text(505,510,'×1',28,600); d.text(505,940,'×3',28,600)
    d.path('M705 782 V775 H750 V716 H705 V669')
    d.path('M705 352 V345 H750 V266 H818 V252',False,dash='5 5')
    d.box(62,1190,256,58,'Vocabulary Embedding',PURPLE,size=17)
    d.line(cx,1190,cx,1039)
    d.box(445,1190,875,58,'N-gram Embedding Layer\n仅第 2 层',PURPLE,size=17)
    d.path('M445 1219 H350 V962 H280')
    d.text(cx,1310,'Input Tokens',17)
    d.line(cx,1288,cx,1248); d.path('M190 1274 H880 V1248')
    notes(d,44,1355,['依据官方小红书第 2 张配图 / 技术报告 Figure 1 重绘；左栏的省略号延续原图表示。',
        'L = 48，Hybrid Block × L/4 = ×12；×N 只表示结构重复，各层参数独立。虚线表示结构展开关联；未展开 Vision Encoder。'],14)
    return d.save(SLUG+'-replica.svg')


def stage(d,x,y,label,sub,fill,w=310):
    d.box(x,y,w,76,fill=fill,stroke='#99aeb7',r=8)
    d.text(x+w/2,y+29,label,18,600)
    d.text(x+w/2,y+55,sub,14)


def improved():
    d = Drawing(1540,1810,'Qwen4 架构预览 · 48 层展开 · 首组与 N-gram Embedding Layer 接入位置')
    heading(d,'02 完整层改进 · 第一组单列，后续十一组明确重复 · 层编号从 1 开始',1540)
    cx=230
    d.box(72,138,205,44,'Prediction Head','#fbefcf',size=17)
    d.box(320,138,235,44,'MTP Modules','#fbefcf',size=17)
    d.box(172,216,116,34,'GR Read',READ,size=16)
    d.path('M230 216 V200 H174 V182')
    d.path('M230 272 H438 V182')
    d.text(438,298,'辅助模块，不计入 48 层',13,color=MUTED)
    d.box(42,310,510,362,fill='#fafcfd',stroke='#95adb8',dash='6 5',r=14)
    d.text(65,343,'第 2–12 组 · 第 5–48 层',21,600,'start')
    stage(d,75,380,'QSA Layer','×1 · 每组最后一层',BLUE)
    stage(d,75,535,'GDN Layer','×3 · 每组前三层',GREEN)
    d.line(cx,535,cx,456); d.path('M230 380 V367 H24 V285 H230 V250')
    d.bracket(422,365,625,'×11')
    d.text(65,650,'每个框均为完整层：Attention 与 MoE 各有 GR。',14,anchor='start',color=MUTED)
    d.box(42,720,510,540,fill='#fafcfd',stroke='#95adb8',dash='6 5',r=14)
    d.text(65,754,'第 1 组 · 第 1–4 层',21,600,'start')
    stage(d,75,790,'QSA Layer','第 4 层',BLUE)
    stage(d,75,905,'GDN Layer','第 3 层',GREEN)
    stage(d,75,1020,'GDN Layer','第 2 层 · 单独接入额外特征',PURPLE)
    stage(d,75,1135,'GDN Layer','第 1 层',GREEN)
    d.path('M230 790 V777 H24 V633 H230 V611')
    for low,high in ((905,866),(1020,981),(1135,1096)): d.line(cx,low,cx,high)
    d.bracket(422,781,1224,'×1')
    d.box(58,1320,300,56,'Vocabulary Embedding',PURPLE,size=17)
    d.path('M208 1320 V1290 H230 V1211')
    d.text(208,1427,'Input Tokens',17); d.line(208,1405,208,1376)
    # A separate side route prevents the one-off PLE from looking repeated.
    d.box(615,1275,875,90,fill=PURPLE,stroke='#aa96b6',r=12)
    d.text(640,1310,'N-gram Embedding Layer',22,600,'start')
    d.text(640,1342,'仅第 2 层 · bigrams / trigrams · 51B 参数表，可从主机内存预取',16,anchor='start')
    d.path('M208 1392 H1052 V1365',color='#856396')
    d.path('M615 1320 H577 V1058 H385',color='#856396')
    d.text(594,1185,'仅一次',14,color='#856396')
    # Layer internals follow the original Figure 1 vocabulary verbatim.
    d.text(615,145,'一个完整层内部：GR Read → 模块 → GR Write',21,600,'start',TEAL)
    detail(d,615,180,'QSA Layer',875)
    detail(d,615,615,'GDN Layer',875)
    d.text(1370,211,'共 12 层',17,600)
    d.text(1370,646,'共 36 层',17,600)
    notes(d,1335,292,['QSA Layer', '使用', 'Qwen Sparse Attn', '', 'GDN Layer', '使用', 'Gated DeltaNet'],14)
    d.box(615,1040,875,195,fill='#f7fafb',stroke='#a9bbc3',r=12)
    d.text(640,1074,'Expanded Residual · 四条分支，逐层保留',20,600,'start',TEAL)
    notes(d,640,1105,['GR Read：先对每条分支做 RMSNorm，再按元素门控并汇成 Hidden States。',
        'GR Write：把模块输出按每条分支的门控系数，加回原有 Expanded Residual。',
        '每个 Attention 与 MoE 分别使用自己的 GR Read / GR Write。',
        'GR Read 已包含归一化；这里不再额外串联一层 pre-normalization。'],15)
    d.box(615,1395,875,120,fill='#fff9f5',stroke='#d6c7bd',r=12)
    d.text(640,1428,'MoE · 每层都有',20,600,'start')
    notes(d,640,1460,['512 个 routed experts 中选择 10 个，另有 1 个 shared expert。',
        'Hidden States 维度 2,560；每个 expert 的 intermediate size 为 640。'],15)
    d.text(44,1480,'48 = (1 + 1 + 1 + 1) + (3 + 1) ×11',22,600,'start',TEAL)
    d.text(44,1553,'逐层核对 · 绿色 GDN Layer，蓝色 QSA Layer；紫色边框标出第 2 层',17,600,'start')
    for i in range(48):
        row,col=divmod(i,24);x=44+col*61;y=1580+row*65
        d.box(x,y,52,42,str(i+1).zfill(2),BLUE if (i+1)%4==0 else GREEN,
              stroke='#8b5c9e' if i==1 else '#a2b1b9',r=5,size=16)
    notes(d,44,1734,['36 个 GDN Layer + 12 个 QSA Layer = 48 个主干层；每层含 MoE。MTP Modules 在主干重复范围之外。',
        '依据官方图与固定版本配置重绘；未展开 Vision Encoder 与 Attention 算子内部。×N 为结构重复，各层参数独立。'],15)
    return d.save(SLUG+'-improved.svg')


def build():
    assert C['num_hidden_layers']==48 and C['hc_count']==4
    assert C['layer_types']==(['linear_attention']*3+['full_attention'])*12
    assert C['ple_layer_ids']==[2] and C['num_experts']==512 and C['num_experts_per_tok']==10
    assert C['shared_expert_intermediate_size']==640
    t=(OUT/'template.html').read_text()
    for key,value in {'REPLICA':replica(),'IMPROVED':improved(),'CONFIG':S['config'], 'REVISION':S['revision'][:12]}.items():
        t=t.replace('{{'+key+'}}',value)
    assert '{{' not in t
    (OUT/'index.html').write_text(t)
    print('Built Qwen4 architecture preview: 2 SVGs and HTML viewer')


if __name__=='__main__':
    build()
