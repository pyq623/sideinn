from pathlib import Path
from html import escape
ROOT=Path(__file__).parent
INK="#30373d"
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
