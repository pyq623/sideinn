(()=>{
 'use strict';
 const KEY='sideinn:hot100:review:v1',valid=new Set(['new','review','done']);
 let records={};
 try{const data=JSON.parse(localStorage.getItem(KEY)||'{}');if(data&&typeof data==='object'&&!Array.isArray(data))records=data;}catch{/* Reading remains available when storage is disabled. */}
 const status=id=>valid.has(records[id])?records[id]:'new';
 const select=document.getElementById('review-status');
 if(select){
  const id=document.body.dataset.problemId;select.value=status(id);
  select.addEventListener('change',()=>{records[id]=select.value;try{localStorage.setItem(KEY,JSON.stringify(records));document.getElementById('save-state').textContent='已保存到当前浏览器';}catch{document.getElementById('save-state').textContent='无法保存，请检查浏览器存储设置';}});
 }
 const search=document.getElementById('question-search');if(!search)return;
 const topic=document.getElementById('topic-filter'),filter=document.getElementById('status-filter');
 const items=[...document.querySelectorAll('.problem-list li')],sections=[...document.querySelectorAll('.topic')];
 const labels={new:'',review:'待巩固',done:'已掌握'};
 for(const item of items){if(item.dataset.kind==='problem')item.querySelector('.status').textContent=labels[status(item.dataset.id)];}
 function render(){
  const query=search.value.trim().toLocaleLowerCase();let problems=0,tips=0;
  for(const item of items){
   const isProblem=item.dataset.kind==='problem';
   const show=(!query||item.dataset.search.toLocaleLowerCase().includes(query))&&(!topic.value||topic.value===item.dataset.topic)&&(!filter.value||(isProblem&&status(item.dataset.id)===filter.value));
   item.hidden=!show;if(show){if(isProblem)problems++;else tips++;}
  }
  for(const section of sections)section.hidden=![...section.querySelectorAll('li')].some(item=>!item.hidden);
  document.getElementById('result-count').textContent=`${problems} 道题 · ${tips} 篇技巧总结`;
  document.getElementById('review-count').textContent=`已掌握 ${items.filter(i=>i.dataset.kind==='problem'&&status(i.dataset.id)==='done').length} / 96`;
  document.getElementById('empty').hidden=problems+tips>0;
  // Hide jumps to filtered-out sections rather than leaving dead navigation.
  for(const link of document.querySelectorAll('.topic-links a'))link.hidden=document.getElementById(link.hash.slice(1)).hidden;
 }
 search.addEventListener('input',render);topic.addEventListener('change',render);filter.addEventListener('change',render);
 document.getElementById('clear-filters').addEventListener('click',()=>{search.value='';topic.value='';filter.value='';render();search.focus();});
 window.addEventListener('pageshow',()=>{try{const data=JSON.parse(localStorage.getItem(KEY)||'{}');if(data&&typeof data==='object'&&!Array.isArray(data))records=data;}catch{}for(const item of items){if(item.dataset.kind==='problem')item.querySelector('.status').textContent=labels[status(item.dataset.id)];}render();});
 render();
})();
