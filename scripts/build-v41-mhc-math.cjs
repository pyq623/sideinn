// Build-time only: NODE_PATH=<mathjax-full@3.2.2 installation>/node_modules node scripts/build-v41-mhc-math.cjs
// Keep mathematical notation as TeX in the SVG template and HTML data-tex spans.
const fs=require('node:fs');
const path=require('node:path');
const {mathjax}=require('mathjax-full/js/mathjax.js');
const {TeX}=require('mathjax-full/js/input/tex.js');
const {SVG}=require('mathjax-full/js/output/svg.js');
const {liteAdaptor}=require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler}=require('mathjax-full/js/handlers/html.js');
const adaptor=liteAdaptor();RegisterHTMLHandler(adaptor);
const math=mathjax.document('',{InputJax:new TeX(),OutputJax:new SVG({fontCache:'none'})});
const root=path.resolve(__dirname,'..');
const esc=s=>s.replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;');
function render(tex){
 const out=adaptor.outerHTML(math.convert(tex,{display:false}));
 if(out.includes('data-mjx-error'))throw new Error(`Invalid TeX: ${tex}`);
 const raw=out.match(/<svg\b[\s\S]*<\/svg>/)[0];
 const box=raw.match(/viewBox="([^"]+)"/)[1].split(' ').map(Number);
 const inner=raw.slice(raw.indexOf('>')+1,raw.lastIndexOf('</svg>'));
 return {raw,box,inner};
}
for(const name of ['single-pass-mhc','mhc-single-pass-comparison','engram-architecture','csa2-flow']){
const template=fs.readFileSync(path.join(root,`assets/v41-architecture/${name}.svg.in`),'utf8');
const svg=template.replace(/<g (data-tex="[^"]*"[^>]+)\/>/g,(_,attrs)=>{
 const a=Object.fromEntries([...attrs.matchAll(/data-([\w-]+)="([^"]*)"/g)].map(m=>[m[1],m[2].replace(/&#x([0-9a-f]+);/gi,(_,n)=>String.fromCodePoint(parseInt(n,16))).replaceAll('&amp;','&').replaceAll('&quot;','"')]));
 const {box,inner}=render(a.tex),size=Number(a.size),scale=size/1000;
 const width=box[2]*scale,height=box[3]*scale;
 const labelWidth=[...a.label].reduce((sum,c)=>sum+(c.charCodeAt(0)>255?1:c===' '?0.3:0.55)*size,0);
 const total=width+labelWidth,x=Number(a.x)-(a.anchor==='middle'?total/2:0),y=Number(a.y)+box[1]*scale;
 return `<g data-tex="${esc(a.tex)}" color="${a.color}" aria-label="${esc(a.tex)}"><svg x="${x}" y="${y}" width="${width}" height="${height}" viewBox="${box.join(' ')}">${inner}</svg><text x="${x+width}" y="${a.y}" style="font-size:${size}px;fill:${a.color}">${esc(a.label)}</text></g>`;
});
fs.writeFileSync(path.join(root,`assets/v41-architecture/${name}.svg`),svg);
}
for(const file of ['v41-single-pass-mhc','v41-engram','v41-csa2']){
const page=path.join(root,`notes/${file}.html`);
const html=fs.readFileSync(page,'utf8').replace(/<span class="mhc-math" data-tex="([^"]*)">[\s\S]*?<\/span>/g,(_,tex)=>{
 const {raw}=render(tex);
 return `<span class="mhc-math" data-tex="${tex}" role="math" aria-label="${esc(tex)}">${raw}</span>`;
});
fs.writeFileSync(page,html);
}
const csaPage=path.join(root,'notes/v41-csa2.html');
fs.writeFileSync(csaPage,fs.readFileSync(csaPage,'utf8').replace(/<!-- CSA_DIAGRAM_START -->[\s\S]*?<!-- CSA_DIAGRAM_END -->/,'<!-- CSA_DIAGRAM_START -->'+fs.readFileSync(path.join(root,'assets/v41-architecture/csa2-flow.svg'),'utf8')+'<!-- CSA_DIAGRAM_END -->'));
console.log('Rendered mHC, Engram and CSA2 math from TeX to standalone SVG paths.');
