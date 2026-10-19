'use strict';
document.documentElement.classList.add('js');
const DATA=JSON.parse(document.getElementById('quiz-data').textContent);
const UNIT=DATA.unit,QS=DATA.questions;
const suffix=new URLSearchParams(location.search).get('preview')==='1'?'-preview':'';
const KEY=DATA.storage_key+suffix;
let state={},storageOK=true;
try{const saved=StudyRecords.read(localStorage,DATA,QS.map(q=>q.id),suffix);state=saved.records;storageOK=saved.ok;}catch(e){storageOK=false;}
const cards=[...document.querySelectorAll('.question')],scope=document.getElementById('scope');
const queryId=id=>document.getElementById('q-'+id), get=id=>state[id]||{};
const valid=(q,arr)=>Array.isArray(arr)?[...new Set(arr)].filter(x=>'ABCD'.includes(x)&&x.length===1).sort():[];
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function save(){try{storageOK=StudyRecords.write(localStorage,KEY,state);}catch(e){storageOK=false;}stats();}
function stats(){
 const submitted=QS.filter(q=>get(q.id).submitted),correct=submitted.filter(q=>get(q.id).last===true),sure=correct.filter(q=>get(q.id).confidence==='sure');
 const first=QS.filter(q=>typeof get(q.id).first==='boolean'),firstOK=first.filter(q=>get(q.id).first);
 document.getElementById('summary').textContent='本轮已提交 '+submitted.length+'/'+QS.length+' · 答对 '+correct.length+' · 答对且自评确定 '+sure.length+' · 首次作答 '+firstOK.length+'/'+first.length+' 正确';
 document.getElementById('save-status').textContent=storageOK?'记录保存在当前浏览器。各部分的记录独立保存；更换设备或清理浏览器前请导出。':'当前浏览器无法持久保存，请在关闭页面前导出记录。';
}
function applyFilter(){for(const q of QS){const s=get(q.id);queryId(q.id).hidden=!(scope.value==='all'||scope.value==='wrong'&&s.last===false||scope.value==='unanswered'&&!s.submitted||scope.value==='uncertain'&&s.submitted&&s.confidence!=='sure');}}
function note(q){const s=get(q.id);queryId(q.id).querySelector('.attempt-note').textContent=s.attempts?'已提交 '+s.attempts+' 次 · 首次 '+(s.first?'答对':'待复习')+' · 最近提交 '+(s.last?'答对':'待复习'):'';}
function feedback(q){const s=get(q.id),card=queryId(q.id);card.querySelector('.feedback').innerHTML=s.submitted?'<div class="result '+(s.last?'':'wrong')+'"><strong>'+(s.last?'本次答对':'本题需要回炉')+' · 你的答案：'+escape(valid(q,s.selected).join(''))+'</strong><p class="meta">把握：'+({sure:'确定，能解释',unsure:'犹豫',guess:'猜测'}[s.confidence]||'未填写')+'。答对后仍请检查每个错误选项的理由。</p>'+q.answer_html+'</div>':'';note(q);}
for(const q of QS){const card=queryId(q.id),s=get(q.id);for(const input of card.querySelectorAll('input'))input.checked=valid(q,s.selected).includes(input.value);card.querySelector('.confidence').value=['sure','unsure','guess'].includes(s.confidence)?s.confidence:'';feedback(q);}
document.getElementById('questions').addEventListener('change',e=>{
 const card=e.target.closest('.question');if(!card)return;const id=card.dataset.id,q=QS.find(q=>q.id===id);
 state[id]={...get(id),selected:[...card.querySelectorAll('input:checked')].map(x=>x.value).sort(),confidence:card.querySelector('.confidence').value,submitted:false};
 feedback(q);save();
});
document.getElementById('questions').addEventListener('click',e=>{
 const submit=e.target.closest('[data-submit]'),retry=e.target.closest('[data-retry]');if(!submit&&!retry)return;
 const id=(submit||retry).dataset[submit?'submit':'retry'],q=QS.find(q=>q.id===id),card=queryId(id),s=get(id);
 if(retry){state[id]={...s,selected:[],confidence:'',submitted:false};for(const input of card.querySelectorAll('input'))input.checked=false;card.querySelector('.confidence').value='';feedback(q);save();return;}
 const picked=[...card.querySelectorAll('input:checked')].map(x=>x.value).sort();
 if(!picked.length||!card.querySelector('.confidence').value){card.querySelector('.feedback').innerHTML='<p class="error" role="alert">请先选择答案，并填写作答把握。</p>';return;}
 const correct=picked.join('')===q.answer;
 state[id]={...s,selected:picked,confidence:card.querySelector('.confidence').value,submitted:true,first:typeof s.first==='boolean'?s.first:correct,last:correct,attempts:(Number(s.attempts)||0)+1};
 feedback(q);save();
});
scope.addEventListener('change',applyFilter);
document.getElementById('export').addEventListener('click',()=>{
 const blob=new Blob([JSON.stringify({unit:UNIT,exportedAt:new Date().toISOString(),records:state},null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='AI笔记-'+(DATA.label||(UNIT===8?'工程专题':'单元'+UNIT))+'-答题记录.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
stats();applyFilter();
