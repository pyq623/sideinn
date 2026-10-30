export const stages=[
 {id:'input',name:'准备',title:'先固定一个查询位置',text:'假设当前处理位置 9。位置 1–8 已经形成四组完整的相邻对；我们跟着位置 9，看它怎样读取历史。',tip:'位置是序列编号；每个方块携带隐藏状态向量。'},
 {id:'compress',name:'压缩',title:'两位置汇成一条全局记忆',text:'Compressor 从每个隐藏状态投影出内容和门控分数。在相邻两位置内，对分数做 softmax，再逐维加权汇合内容。归一化等处理后，四组输入形成四条 main KV 条目。',tip:'压缩的是全局缓存的序列长度；原始 token 和主干隐藏状态仍然存在。'},
 {id:'keys',name:'索引',title:'给每条记忆准备检索键',text:'从压缩内容投影出 indexer K（K 指的是 Key，键）；当前位置的隐藏状态生成本层 indexer Q（Q 指的是 Query，查询）。两者一起用于下一步的检索评分。真实实现从归一化后、RoPE 前的压缩表示生成 indexer K；图中合并展示这些准备工作。',tip:'indexer K 用于找内容；main KV 保存随后实际读取的内容。'},
 {id:'score',name:'评分',title:'先判断哪条记忆值得读',text:'本例 indexer Q 对四条可见记忆评分，依次得到 0.2、0.9、0.4、0.7。这些教学分数用于排序；它们不是主注意力的最终权重。',tip:'这是 Encoder Full，检索范围是因果可见的压缩条目。Decoder 的分层候选池在后续架构步骤另讲。'},
 {id:'select',name:'选址',title:'选出地址，再按地址取内容',text:'Top-2 选择得分最高的条目 2、4。首先得到的是地址列表，再用 Gather 读取对应的第 2、4 条全局 KV。左侧长箭头表示内容来自 main KV 缓存，评分没有变成 KV。',tip:'本例 Top-2 只为看清选择；真实配置是 Top-512。压缩率 2 与 Top-K 的 K 是两个参数。'},
 {id:'attend',name:'读取',title:'主注意力同时看全局和近期',text:'main Q 来自当前位置 9 的本层隐藏状态。它与刚取回的两条全局 KV、局部窗口 7/8/9 的 SWA KV 一起参与注意力计算，得到本层的新输出。SWA 的三个位置由窗口确定，不参加前面的 Top-K 筛选。',tip:'Full 至此完成缓存构造、选址与读取；后续 Reuse 仍会用自己的 main Q 和 SWA 计算新输出。'}
];
// This is a teaching trace, independent of model weights.
export const toyScores=[.2,.9,.4,.7];
export const selectedEntries=toyScores.map((score,i)=>({id:i+1,score})).sort((a,b)=>b.score-a.score).slice(0,2).map(x=>x.id);
const $=id=>document.getElementById(id),root=document.querySelector('.csa-player'),buttons=[...document.querySelectorAll('[data-csa-step]')];
let index=0,timer=null,motionTimer=null;
const pref=matchMedia('(prefers-reduced-motion: reduce)');$('csa-reduce').checked=pref.matches;
function stop(){clearTimeout(timer);timer=null;$('csa-play').textContent='自动播放';$('csa-play').setAttribute('aria-pressed','false');}
function animate(){clearTimeout(motionTimer);root.classList.remove('is-moving');if(!$('csa-reduce').checked){requestAnimationFrame(()=>{root.classList.add('is-moving');motionTimer=setTimeout(()=>root.classList.remove('is-moving'),3100);});}}
function render(i){index=Math.max(0,Math.min(stages.length-1,i));const s=stages[index];root.classList.add('is-ready');root.classList.toggle('reduce-motion',$('csa-reduce').checked);$('csa-phase').textContent=`0${index+1} / 06 · ${s.name}`;$('csa-step-title').textContent=s.title;$('csa-narration').textContent=s.text;$('csa-takeaway').textContent=s.tip;$('csa-compression').hidden=index!==1;$('csa-deepinfer').hidden=index!==5;$('csa-mobile-caption').textContent=`${index+1} / 6 · ${s.title}。${s.text}`;$('csa-prev').disabled=index===0;$('csa-next').disabled=index===5;buttons.forEach((b,n)=>{if(n===index)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});document.querySelectorAll('[data-part]').forEach(g=>{const n=Number(g.dataset.part);g.classList.toggle('is-current',n===index);g.classList.toggle('is-future',n>index);});document.querySelectorAll('[data-entry]').forEach(e=>e.classList.toggle('csa-selected',index>=4&&selectedEntries.includes(Number(e.dataset.entry))));$('csa-status').textContent=`${timer?'自动播放':'手动阅读'} · ${index+1} / 6`;history.replaceState(null,'',`#${s.id}`);animate();}
function schedule(){timer=setTimeout(()=>{if(index===5){stop();$('csa-status').textContent='演示完成 · 可逐步回看';return;}render(index+1);schedule();},6500);}
buttons.forEach(b=>b.addEventListener('click',()=>{stop();render(Number(b.dataset.csaStep));}));
$('csa-prev').onclick=()=>{stop();render(index-1);};$('csa-next').onclick=()=>{stop();render(index+1);};$('csa-replay').onclick=()=>{stop();render(index);};
$('csa-play').onclick=()=>{if(timer){stop();$('csa-status').textContent=`已暂停 · ${index+1} / 6`;return;}if(index===5)render(0);schedule();$('csa-play').textContent='暂停播放';$('csa-play').setAttribute('aria-pressed','true');$('csa-status').textContent=`自动播放 · ${index+1} / 6`;};
$('csa-reduce').onchange=()=>{root.classList.toggle('reduce-motion',$('csa-reduce').checked);};
addEventListener('hashchange',()=>{stop();render(Math.max(0,stages.findIndex(s=>`#${s.id}`===location.hash)));});
document.addEventListener('visibilitychange',()=>{if(document.hidden){stop();clearTimeout(motionTimer);root.classList.remove('is-moving');$('csa-status').textContent=`已暂停 · ${index+1} / 6`;}});
render(Math.max(0,stages.findIndex(s=>`#${s.id}`===location.hash)));

// Keep the user's preferred width while adapting the actual width to the viewport.
const layout=$('csa-layout'),splitter=$('csa-splitter');
const widthKey='sideinn:v41-csa2:sidebar-width',defaultWidth=240,minWidth=180;
const desktop=matchMedia('(min-width: 901px)');
let preferredWidth=defaultWidth,drag=null;
try{const saved=Number(localStorage.getItem(widthKey));if(Number.isFinite(saved)&&saved>=minWidth&&saved<=420)preferredWidth=saved;}catch{}
function maxWidth(){return Math.max(minWidth,Math.min(420,Math.floor(layout.clientWidth*.45)));}
function applyWidth(){
 if(!desktop.matches)return;
 const width=Math.round(Math.max(minWidth,Math.min(maxWidth(),preferredWidth)));
 layout.style.setProperty('--csa-sidebar-width',`${width}px`);
 splitter.setAttribute('aria-valuenow',String(width));splitter.setAttribute('aria-valuemax',String(maxWidth()));splitter.setAttribute('aria-valuetext',`${width} 像素`);
}
function setWidth(width){preferredWidth=Math.round(Math.max(minWidth,Math.min(maxWidth(),width)));applyWidth();try{localStorage.setItem(widthKey,String(preferredWidth));}catch{}}
function resetWidth(){setWidth(defaultWidth);}
function endDrag(){drag=null;layout.classList.remove('is-resizing');}
splitter.addEventListener('pointerdown',event=>{
 if(!desktop.matches||event.button!==0)return;
 event.preventDefault();splitter.focus();drag={pointer:event.pointerId,start:event.clientX,width:$('csa-explanation').getBoundingClientRect().width};
 splitter.setPointerCapture(event.pointerId);layout.classList.add('is-resizing');
});
splitter.addEventListener('pointermove',event=>{if(drag&&drag.pointer===event.pointerId)setWidth(drag.width+event.clientX-drag.start);});
splitter.addEventListener('pointerup',event=>{if(drag&&drag.pointer===event.pointerId){splitter.releasePointerCapture(event.pointerId);endDrag();}});
splitter.addEventListener('pointercancel',endDrag);splitter.addEventListener('lostpointercapture',endDrag);
splitter.addEventListener('dblclick',resetWidth);
splitter.addEventListener('keydown',event=>{
 const current=Number(splitter.getAttribute('aria-valuenow')),step=event.shiftKey?25:10;
 const widths={ArrowLeft:current-step,ArrowRight:current+step,Home:minWidth,End:maxWidth()};
 if(event.key in widths){event.preventDefault();setWidth(widths[event.key]);}
});
$('csa-reset-width').addEventListener('click',resetWidth);
new ResizeObserver(applyWidth).observe(layout);
desktop.addEventListener('change',()=>{endDrag();applyWidth();});
applyWidth();
