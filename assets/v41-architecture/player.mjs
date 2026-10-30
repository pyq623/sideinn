import {steps,regions} from './steps.mjs?v=20261030-candidate-topk';
import {mountCacheView} from './cache-view.mjs?v=20261030-candidate-topk';
const $=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';
const colors={hidden:'#ac651f',cache:'#207b5b',index:'#327aaf',memory:'#87549d'};
const renderCache=mountCacheView($('global-kv-map'));
let index=0,focus=matchMedia('(max-width:700px)').matches,reduced=matchMedia('(prefers-reduced-motion:reduce)').matches;
let routes=[],elapsed=0,last=0,raf=0,running=false,automatic=false;
const segmentMs=1600,holdMs=8500;
function text(id,value){if($(id).textContent!==value)$(id).textContent=value;}
const animationDuration=()=>reduced?0:routes.length*segmentMs;
function svg(tag,attrs){const e=document.createElementNS(NS,tag);for(const [key,val] of Object.entries(attrs))e.setAttribute(key,val);return e;}
for(const [i,step] of steps.entries()){const option=document.createElement('option');option.value=String(i);option.textContent=`${String(i+1).padStart(2,'0')} · ${step.title}`;$('step-select').append(option);}
for(let i=0;i<40;i++){const el=document.createElement('span');el.className='layer'+(i>=20?' decoder':'');el.textContent=String(i);el.title=`${i>=20?'Decoder':'Encoder'} 层 ${i}`;$('layer-strip').append(el);}
$('reduce').checked=reduced;
function view(){const step=steps[index];$('diagram').setAttribute('viewBox',(focus?step.view:[-8,-8,926,485]).join(' '));$('focus').setAttribute('aria-pressed',String(focus));$('focus').textContent=focus?'显示完整原图':'聚焦当前区域';}
function stop(){cancelAnimationFrame(raf);raf=0;running=false;last=0;}
function paint(){
 const duration=animationDuration();
 routes.forEach(({path,dot,length},i)=>{
  const fraction=reduced?1:Math.max(0,Math.min(1,(elapsed-i*segmentMs)/segmentMs));
  path.style.strokeDashoffset=String(length*(1-fraction));
  const point=path.getPointAtLength(length*fraction);dot.setAttribute('cx',point.x);dot.setAttribute('cy',point.y);dot.style.opacity=fraction===0?'0':'1';
 });
 text('auto',automatic&&running?'暂停':'自动播放');
 $('pause').disabled=elapsed>=duration;
 text('pause',running?'暂停动作':'继续动作');
 text('status',automatic&&running?(elapsed<duration?'自动播放 · 正在沿连线演示':'自动播放 · 停留阅读后进入下一步'):running?'本步动作演示中 · 完成后停住':elapsed<duration?'动作已暂停':'本步已完成 · 可手动下一步或重播');
}
function frame(now){
 if(!running)return;
 if(last)elapsed+=(now-last)*Number($('speed').value);last=now;
 paint();
 if(elapsed>=animationDuration()+(automatic?holdMs:0)){
  if(automatic&&index<steps.length-1){render(index+1,{animate:true,auto:true});return;}
  automatic=false;stop();paint();return;
 }
 raf=requestAnimationFrame(frame);
}
function start(){if(running)return;running=true;last=0;raf=requestAnimationFrame(frame);paint();}
const cedRelated={
 hidden:{text:'CED（Causal Encoder–Decoder，因果编码器—解码器）从 Encoder 的最终表示分出两条路径。对应 DeepInfer 的“得到完整的第三层输出”。',hash:'ced-output',label:'看 Encoder 输出与分流动画 →',note:'section-3-1',noteLabel:'阅读 §3.1：CED 两路结构 →'},
 'global-path':{text:'这就是 DeepInfer 的第一条路：全部位置投影成 Decoder 的全局 KV，提供历史内容；隐藏状态仍可沿另一条路继续前向。',hash:'fork',label:'看第一条路：全长 KV 投影 →',note:'section-3-1',noteLabel:'阅读 §3.1：CED 两路结构 →'},
 'tail-path':{text:'对应 DeepInfer 的第二条路：尾部隐藏状态继续经过 Decoder，计算本层 Q 和局部 SWA KV。只保留短后缀，是 CED 配合 bounded replay（有限重放）的优化。',hash:'tail',label:'看第二条路：尾部继续前向 →',note:'section-4-2',noteLabel:'阅读 §4.2：两种重放的作用 →'}
};
function render(i,{animate=true,auto=false}={}){
 stop();automatic=auto;index=Math.max(0,Math.min(steps.length-1,i));const step=steps[index];
 document.querySelector('.explain').scrollTop=0;
 $('phase').textContent=step.phase;$('step-title').textContent=step.title;$('narration').textContent=step.text;$('takeaway').textContent=step.takeaway;
 $('mhc-card').hidden=step.id!=='input';
 $('mhc-link').hidden=step.id!=='input';
 $('engram-card').hidden=step.id!=='engram';
 $('engram-link').hidden=step.id!=='engram';
 $('csa-card').hidden=step.id!=='encoder-full';
 $('csa-link').hidden=step.id!=='encoder-full';
 $('candidate-card').hidden=step.id!=='candidate';
 $('candidate-link').hidden=step.id!=='candidate';
 const ced=cedRelated[step.id];$('ced-related').hidden=!ced;$('ced-link').hidden=!ced;
 if(ced){$('ced-related-text').textContent=ced.text;$('ced-animation').href=`./deepinfer-walkthrough.html#${ced.hash}`;$('ced-animation').textContent=ced.label;$('ced-link').href=$('ced-animation').href;$('ced-note').href=`./deepinfer.html#${ced.note}`;$('ced-note').textContent=ced.noteLabel;}

 $('mobile-summary').textContent=`${index+1} / ${steps.length} · ${step.phase}。${step.takeaway}`;
 $('diagram-desc').textContent=`第 ${index+1} 步：${step.title}。${step.text}`;
 $('count').textContent=`${index+1} / ${steps.length}`;$('step-select').value=String(index);
 $('prev').disabled=index===0;$('next').disabled=index===steps.length-1;
 $('source').href=`https://arxiv.org/html/2609.19969v1#${step.source}`;
 $('cache-count').textContent=`Encoder ${step.cache[0]} 套 · Decoder ${step.cache[1]} 套`;
 renderCache(step);
 for(const [i,el] of [...$('layer-strip').children].entries())el.classList.toggle('active',step.layers.includes(i));
 const phase=index<8?0:index<11?1:index<19?2:3;
 [...document.querySelectorAll('.chapters button')].forEach((el,i)=>el.setAttribute('aria-current',String(i===phase)));
 $('highlights').replaceChildren();$('routes').replaceChildren();routes=[];
 for(const key of step.focus){const [x,y,w,h]=regions[key];$('highlights').append(svg('rect',{x:x-2,y:y-2,width:w+4,height:h+4,rx:5,class:'highlight'}));}
 for(const item of step.paths){
  const path=svg('path',{d:item.d,stroke:colors[item.kind],class:'route'}),dot=svg('circle',{r:4.5,fill:colors[item.kind],class:'dot'});
  $('routes').append(path,dot);const length=path.getTotalLength();path.style.strokeDasharray=String(length);routes.push({path,dot,length});
 }
 view();elapsed=animate?0:animationDuration();paint();
 history.replaceState(null,'',`#${step.id}`);
 if(animate&&(routes.length||auto))start();
}
$('prev').addEventListener('click',()=>render(index-1));$('next').addEventListener('click',()=>render(index+1));
$('replay').addEventListener('click',()=>render(index));
$('auto').addEventListener('click',()=>{
 if(automatic&&running){automatic=false;stop();paint();return;}
 if(index===steps.length-1){render(0,{animate:true,auto:true});return;}
 automatic=true;if(elapsed>=animationDuration())elapsed=0;start();
});
$('pause').addEventListener('click',()=>{if(running){automatic=false;stop();paint();}else start();});
$('step-select').addEventListener('change',e=>render(Number(e.target.value)));
for(const btn of document.querySelectorAll('[data-step]'))btn.addEventListener('click',()=>render(steps.findIndex(s=>s.id===btn.dataset.step)));
$('focus').addEventListener('click',()=>{focus=!focus;view();});
$('reduce').addEventListener('change',()=>{reduced=$('reduce').checked;if(reduced){elapsed=animationDuration();if(!automatic)stop();}paint();});
const motion=matchMedia('(prefers-reduced-motion:reduce)');motion.addEventListener('change',e=>{reduced=e.matches;$('reduce').checked=reduced;if(reduced){automatic=false;stop();elapsed=animationDuration();}paint();});
document.addEventListener('visibilitychange',()=>{if(document.hidden){automatic=false;stop();paint();}});
window.addEventListener('pagehide',stop);
window.addEventListener('hashchange',()=>{const i=steps.findIndex(s=>`#${s.id}`===location.hash);if(i>=0&&i!==index)render(i,{animate:false});});
window.addEventListener('keydown',e=>{if(e.target.closest('input,select,button,textarea,a,.mhc-inline-scroll'))return;if(e.key==='ArrowRight'){e.preventDefault();render(index+1);}else if(e.key==='ArrowLeft'){e.preventDefault();render(index-1);}else if(e.code==='Space'){e.preventDefault();if(running){automatic=false;stop();paint();}else if(elapsed<animationDuration())start();else $('auto').click();}});
const initial=steps.findIndex(s=>`#${s.id}`===location.hash);render(initial>=0?initial:0,{animate:false});
