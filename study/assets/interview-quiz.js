'use strict';
document.documentElement.classList.add('js');
const DATA=JSON.parse(document.getElementById('quiz-data').textContent), UNIT=DATA.unit, QS=DATA.questions;
const PREVIEW=new URLSearchParams(location.search).get('preview')==='1';
const KEY='accio-agentic-rl-v1-u'+UNIT+(PREVIEW?'-preview':'');
let state={records:{},oral:{}},storageOK=true;
try {const s=JSON.parse(localStorage.getItem(KEY)||'null');if(s&&s.schema===1&&s.records&&typeof s.records==='object'&&!Array.isArray(s.records))state={records:s.records,oral:s.oral||{}};}catch(e){storageOK=false;}
const $=id=>document.getElementById(id), rec=id=>state.records[id]||{};
const safe=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function save(){try{localStorage.setItem(KEY,JSON.stringify({schema:1,...state}));storageOK=true;}catch(e){storageOK=false;}stats();}
function stats(){
 const done=QS.filter(q=>typeof rec(q.id).last==='boolean'), correct=done.filter(q=>rec(q.id).last), first=done.filter(q=>rec(q.id).first);
 $('summary').textContent='已练 '+done.length+'/'+QS.length+' · 最近答对 '+correct.length+' · 首次答对 '+first.length+'/'+done.length+' · '+(PREVIEW?'预览练习（独立记录）':'正式练习');
 $('save-status').textContent=storageOK?'答题与口述笔记保存在当前浏览器；换设备或清理浏览器前请导出。':'浏览器当前无法保存，关闭页面前请导出。';
}
function filter(){
 const mode=$('scope').value;
 for(const q of QS){const s=rec(q.id);$('q-'+q.id).hidden=!(mode==='all'||mode==='wrong'&&s.last===false||mode==='unanswered'&&typeof s.last!=='boolean'||mode==='uncertain'&&typeof s.last==='boolean'&&s.lastConfidence!=='sure');}
 for(const lesson of document.querySelectorAll('.lesson')){
  const visible=[...lesson.querySelectorAll('.question')].some(c=>!c.hidden);
  lesson.hidden=mode!=='all'&&!visible;
 }
 const visible=QS.filter(q=>!$('q-'+q.id).hidden).length;
 $('filter-status').textContent=mode==='all'?'每节练习紧接知识点，答案可返回本节。':visible?'当前显示 '+visible+' 道；重做时保留上次判定，提交后更新筛选。':'当前筛选没有题目，可切换“全部题目”。';
}
function feedback(q){
 const s=rec(q.id),card=$('q-'+q.id);
 card.querySelector('.attempt-note').textContent=s.attempts?'已提交 '+s.attempts+' 次 · 首次'+(s.first?'答对':'待复习')+' · 最近'+(s.last?'答对':'待复习'):'';
 card.querySelector('.feedback').innerHTML=s.show&&typeof s.last==='boolean'?'<div class="result '+(s.last?'':'wrong')+'"><strong>'+(s.last?'本次答对':'本题需要复习')+' · 你的答案：'+safe((s.lastSelected||[]).join(''))+'</strong>'+q.answer_html+'</div>':'';
}
for(const q of QS){
 const card=$('q-'+q.id),s=rec(q.id), selected=Array.isArray(s.selected)?s.selected:[];
 for(const input of card.querySelectorAll('input'))input.checked=selected.includes(input.value);
 card.querySelector('.confidence').value=['sure','unsure','guess'].includes(s.confidence)?s.confidence:'';feedback(q);
}
for(const t of document.querySelectorAll('[data-oral]')){t.value=typeof state.oral[t.dataset.oral]==='string'?state.oral[t.dataset.oral]:'';t.addEventListener('input',()=>{state.oral[t.dataset.oral]=t.value;save();});}
$('course-content').addEventListener('change',e=>{
 const card=e.target.closest('.question');if(!card)return;
 const q=QS.find(q=>q.id===card.dataset.id);if(!q)return;
 state.records[q.id]={...rec(q.id),selected:[...card.querySelectorAll('input:checked')].map(x=>x.value).sort(),confidence:card.querySelector('.confidence').value,show:false};feedback(q);save();
});
$('course-content').addEventListener('click',e=>{
 const btn=e.target.closest('[data-submit],[data-retry]');if(!btn)return;
 const id=btn.dataset.submit||btn.dataset.retry,q=QS.find(q=>q.id===id);if(!q)return;
 const card=$('q-'+id),s=rec(id);
 if(btn.dataset.retry){
  state.records[id]={...s,selected:[],confidence:'',show:false};for(const input of card.querySelectorAll('input'))input.checked=false;card.querySelector('.confidence').value='';feedback(q);save();return;
 }
 const selected=[...card.querySelectorAll('input:checked')].map(x=>x.value).sort(),confidence=card.querySelector('.confidence').value;
 if(!selected.length||!confidence){card.querySelector('.feedback').innerHTML='<p role="alert" class="error">请先选择答案，并填写作答把握。</p>';return;}
 if(s.show){return;} // 不重复累计同一次提交
 const correct=selected.join('')===q.answer;
 state.records[id]={...s,selected,confidence,lastSelected:selected,lastConfidence:confidence,last:correct,first:typeof s.first==='boolean'?s.first:correct,attempts:(Number(s.attempts)||0)+1,show:true};feedback(q);save();filter();
});
$('scope').addEventListener('change',filter);
$('practice-mode').addEventListener('click',()=>{const active=document.body.classList.toggle('practice-only');$('practice-mode').textContent=active?'显示讲义与习题':'只看习题';$('practice-mode').setAttribute('aria-pressed',String(active));});
$('export').addEventListener('click',()=>{
 const blob=new Blob([JSON.stringify({schema:1,course:'accio-agentic-rl',unit:UNIT,exportedAt:new Date().toISOString(),records:state.records,oral:state.oral},null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='AgenticRL-单元'+UNIT+'-答题与笔记.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
stats();filter();
