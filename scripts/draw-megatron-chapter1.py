#!/usr/bin/env python3
"""Build original, editable teaching diagrams for Megatron chapter 1."""
from pathlib import Path
from html import escape
OUT=Path(__file__).resolve().parents[1]/'assets/rl-infra-megatron'
INK='#234e43'; BLUE='#e9effc'; GREEN='#e6f3e9'; GOLD='#fff1cc'; GRAY='#f5f7f5'
class SVG:
 def __init__(self,w,h,title):
  self.w=w;self.h=h;self.s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img"><title>{escape(title)}</title><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0L9 4.5L0 9" fill="{INK}"/></marker></defs><rect width="100%" height="100%" fill="#fff"/><g font-family="Arial, PingFang SC, Microsoft YaHei, sans-serif" fill="{INK}">']
 def text(self,x,y,t,size=22,bold=False,anchor='middle'):
  self.s.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}">{escape(t)}</text>')
 def box(self,x,y,w,h,lines,color=GREEN,size=22):
  self.s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{color}" stroke="#b9cec5" stroke-width="1.5"/>')
  for j,t in enumerate(lines):self.text(x+w/2,y+h/2+(j-(len(lines)-1)/2)*(size+9)+size*.34,t,size)
 def arrow(self,x,y,xx,yy,dash=False):
  self.s.append(f'<path d="M{x} {y} L{xx} {yy}" fill="none" stroke="{INK}" stroke-width="2" marker-end="url(#arrow)"'+(' stroke-dasharray="6 5"' if dash else '')+'/>')
 def line(self,x,y,xx,yy):self.s.append(f'<path d="M{x} {y} L{xx} {yy}" stroke="#cbd9d0" stroke-width="2"/>')
 def save(self,name): (OUT/name).write_text(''.join(self.s)+'</g></svg>')

s=SVG(1000,490,'8 rank 教学进程组')
s.text(500,34,'8 个 rank：r = t + 2d + 4p',27,True)
for stage,x in [(0,20),(1,520)]:
 s.box(x,58,460,310,[],GRAY)
 s.text(x+230,94,f'PP 阶段 {stage}：持有不同的层',24,True)
 for d,y in [(0,119),(1,238)]:
  s.text(x+230,y+19,f'DP 副本 {d} · 样本批次 {d}',20)
  for t,bx in [(0,x+30),(1,x+250)]:
   s.box(bx,y+35,180,62,[f'rank {t+2*d+4*stage} · t={t}'],BLUE if d==0 else GREEN,22)
  s.arrow(x+212,y+65,x+246,y+65)
 s.text(x+230,360,'每一行的两张卡组成 TP 组',21)

for x in [72,428,572,928]: s.arrow(x,216,x,270,True)
s.text(500,404,'TP：同行协作拆层   DP：同阶段、同 t 的两份副本',23,True)
s.text(500,442,'PP 连接：0 ↔ 4，1 ↔ 5，2 ↔ 6，3 ↔ 7',23)
s.text(500,476,'CP = 1；图中距离不表示物理网络拓扑',20)
s.save('process-groups.svg')

s=SVG(1000,760,'两层 MLP 的 TP 前反向')
s.text(500,34,'前向：不同特征分片 → 同一输出的部分和',26,True)
for i,y in [(0,85),(1,212)]:
 s.text(53,y+32,f'卡 {i}',21,True)
 s.box(95,y,130,75,['X','[N,H]'],BLUE)
 s.arrow(225,y+38,255,y+38)
 s.box(260,y,195,75,[f'× A{i} → φ',f'Y{i} [N,F/2]'],GREEN)
 s.arrow(455,y+38,485,y+38)
 s.box(490,y,205,75,[f'× B{i}',f'Z{i} [N,H]'],GOLD)
 s.arrow(695,y+38,745,186)
s.box(750,145,230,83,['all-reduce 求和','Z = Z₀ + Z₁'],BLUE,23)
s.text(500,332,'两卡各得到完整 Z；Y₀ 与 Y₁ 无须先拼接',23,True)
s.line(20,356,980,356)
s.text(500,395,'反向：本地求导 → 汇总两条输入梯度贡献',26,True)
for i,y in [(0,442),(1,568)]:
 s.text(53,y+30,f'卡 {i}',21,True)
 s.box(95,y,130,75,['G = ∂L/∂Z','[N,H]'],BLUE,20)
 s.arrow(225,y+38,255,y+38)
 s.box(260,y,195,75,[f'G B{i}ᵀ → × φ′',f'D{i} [N,F/2]'],GREEN,22)
 s.arrow(455,y+38,485,y+38)
 s.box(490,y,205,75,[f'V{i} = D{i} A{i}ᵀ','[N,H] 部分梯度'],GOLD,21)
 s.arrow(695,y+38,745,540)
s.box(750,501,230,83,['all-reduce 求和','∂L/∂X = V₀ + V₁'],BLUE,21)
s.text(500,697,'每卡还局部计算：∂L/∂Bᵢ = YᵢᵀG；∂L/∂Aᵢ = XᵀDᵢ',23,True)
s.text(500,735,'不启用 SP；G 是同一逻辑损失的梯度，不重复累加两份损失',20)
s.save('mlp-forward-backward.svg')

s=SVG(1000,840,'SP 和 CP 的序列布局')
s.text(500,35,'SP：同一 TP 组内，在 MLP 边界改变布局',26,True)
xs=[15,215,415,615,815]
for x,lines,col in zip(xs,[['序列行分片','每卡 [N/2,H]'],['all-gather X','各卡 [N,H]'],['列并行 + 激活','Yᵢ [N,F/2]'],['行并行','Zᵢ [N,H]'],['reduce-scatter','每卡 [N/2,H]']],[GREEN,BLUE,GREEN,GOLD,GREEN]):
 s.box(x,65,170,98,lines,col,20)
for x in xs[:-1]:s.arrow(x+170,114,x+195,114)
s.text(500,201,'前向：收集不同位置 → 算不同特征 → 求和并分回位置',22,True)
s.box(25,222,950,82,['反向：出口梯度 all-gather → 本地反向 → 输入梯度 reduce-scatter','计算 A 的权重梯度时，若只保存了分片 X，还需重新收集 X'],GRAY,21)
s.line(20,325,980,325)
s.text(500,366,'CP：每卡负责本地 query，仍需其他位置的 KV',26,True)
s.text(500,398,'教学配置 CP=2、TP=1；序列 1–4；因果掩码',21)
s.box(35,423,410,95,['CP rank 0：位置 1、2','本地 Q₁₂、K₁₂、V₁₂'],GREEN)
s.box(555,423,410,95,['CP rank 1：位置 3、4','本地 Q₃₄、K₃₄、V₃₄'],GREEN)
s.arrow(240,518,360,546);s.arrow(760,518,640,546)
s.box(185,550,630,65,['all-gather：两边获得位置 1–4 的 KV'],BLUE,23)
s.arrow(360,615,240,643);s.arrow(640,615,760,643)
s.box(35,648,410,79,['q₁ 读 1；q₂ 读 1、2','只输出位置 1、2 的结果'],GREEN,22)
s.box(555,648,410,79,['q₃ 读 1–3；q₄ 读 1–4','只输出位置 3、4 的结果'],GREEN,22)
s.box(35,749,930,65,['反向：KV 梯度贡献先相加，再按位置归回 0 / 1（reduce-scatter）'],GOLD,22)
s.save('sp-cp-layout.svg')

s=SVG(1000,540,'MoE 路由、分发及返回')
s.text(500,34,'EP=2，ETP=1：4 个专家，每个 token 选 2 个',26,True)
for x,t in [(150,'① 原位置与路由'),(500,'② 专家所在地'),(850,'③ 返回并加权')]:s.text(x,77,t,23,True)
s.box(20,115,260,145,['rank 0：x_a','E0：0.6','E2：0.4'],BLUE,23)
s.box(20,310,260,145,['rank 1：x_b','E1：0.3','E2：0.7'],BLUE,23)
s.box(370,115,260,145,['rank 0','E0 计算 a','E1 计算 b'],GREEN,23)
s.box(370,310,260,145,['rank 1','E2 计算 a、b','E3 本例空闲'],GREEN,23)
s.box(720,115,260,145,['回到 rank 0','o_a = 0.6 E0(x_a)','+ 0.4 E2(x_a)'],GOLD,22)
s.box(720,310,260,145,['回到 rank 1','o_b = 0.3 E1(x_b)','+ 0.7 E2(x_b)'],GOLD,22)
s.arrow(280,155,365,155);s.arrow(280,225,365,345);s.arrow(280,345,365,225);s.arrow(280,415,365,415)
s.arrow(630,155,715,155);s.arrow(630,225,715,345);s.arrow(630,345,715,225);s.arrow(630,415,715,415)
s.text(500,493,'dispatch：按专家重新分布；combine：恢复 token 原位置',23,True)
s.text(500,526,'箭头表示分支依赖；本地分支不必经过网络，专家参数留在原地',20)
s.save('moe-dispatch.svg')
print('Generated 4 teaching diagrams')
