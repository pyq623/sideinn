(()=>{
 'use strict';
 const $=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 let notes=[],category='';
 if(/^#(?:chapter-\d+|section-\d+-\d+)$/.test(location.hash)){location.replace('./notes/rsi.html'+location.hash);return;}
 function render(){
  const query=$('library-search').value.trim().toLocaleLowerCase();
  const visible=notes.filter(n=>(!category||n.category===category)&&(!query||[n.title,n.english,n.summary,n.category,...n.tags].join(' ').toLocaleLowerCase().includes(query)));
  $('result-count').textContent=visible.length+' 篇笔记';
  $('note-list').innerHTML=visible.map(n=>`<a class="note-card" data-note-id="${esc(n.id)}" href="${esc(n.href)}" aria-label="阅读 ${esc(n.title)}"><div class="note-cover" aria-hidden="true"><span class="cover-label">支线 / ${esc(n.coverLabel||n.id.toUpperCase())}</span><strong>${esc(n.coverLabel||n.id.toUpperCase())}</strong><span class="cover-sub">${esc(n.title)}</span><span class="cover-foot">${esc(n.coverFooter||'RESEARCH NOTES')}</span></div><div class="note-content"><div class="note-eyebrow">${esc(n.category)}<span>更新于 ${esc(n.updated.replaceAll('-','.'))}</span></div><h3>${esc(n.title)}</h3><p class="note-english">${esc(n.english)}</p><p class="note-summary">${esc(n.summary)}</p><div class="note-tags">${n.tags.map(t=>`<span>${esc(t)}</span>`).join('')}</div><div class="note-bottom"><span>${esc(n.details||`${n.chapters} 章 · ${n.formulas} 个公式 · ${n.references} 条参考`)}</span><span class="read-note">${esc(n.readLabel||'阅读笔记')} <span aria-hidden="true">↗</span></span></div></div></a>`).join('');
  $('empty-state').hidden=visible.length>0;
  for(const b of $('category-list').querySelectorAll('button')){const active=(b.dataset.category||'')===category;b.setAttribute('aria-pressed',String(active));}
 }
 function categories(){
  const counts=new Map();for(const n of notes)counts.set(n.category,(counts.get(n.category)||0)+1);
  $('category-list').innerHTML=`<button type="button" data-category="" aria-pressed="true"><span>全部笔记</span><span>${notes.length}</span></button>`+[...counts].map(([name,count])=>`<button type="button" data-category="${esc(name)}" aria-pressed="false"><span>${esc(name)}</span><span>${count}</span></button>`).join('');
 }
 $('library-search').addEventListener('input',render);
 $('category-list').addEventListener('click',e=>{const b=e.target.closest('button');if(b){category=b.dataset.category;render();}});
 $('clear-filters').addEventListener('click',()=>{category='';$('library-search').value='';render();$('library-search').focus();});
 async function init(){
  const response=await fetch('./assets/notebook.json');if(!response.ok)throw Error('Unable to load notebook');notes=await response.json();
  $('total-notes').textContent=notes.length+' 篇笔记';categories();render();$('note-list').setAttribute('aria-busy','false');
 }
 init().catch(error=>{$('note-list').setAttribute('aria-busy','false');$('load-status').textContent='笔记目录暂时无法加载，请刷新重试。';console.error(error);});
})();
