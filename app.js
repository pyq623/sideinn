/* Each reader declares its canonical Markdown and formula catalog. */
(()=>{
 'use strict';
 const $=id=>document.getElementById(id),article=$('document');
 const noteId=document.body.dataset.noteId||'rsi',source=document.body.dataset.noteSource||'../notes.md';
 const KEY=noteId==='rsi'?'rsi-study-notes-draft-v1':'study-notes-draft-v1-'+noteId;
 let paragraphEntries=[];
 let base='',text='',catalog=[],renderVersion=0,editing=false,headings=[],observer,searchHits=[],hitIndex=-1,previewTimer,searchTimer,lastRemote='',activeQuery='';
 const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
 const hash=s=>{let h=2166136261;for(let i=0;i<s.length;i++)h=Math.imul(h^s.charCodeAt(i),16777619);return (h>>>0).toString(16);};
 const normalize=s=>s.replace(/\s+/g,'');
 function notice(message){$('notice').textContent=message;$('notice').hidden=!message;}
 function getDraft(){try{return JSON.parse(localStorage.getItem(KEY)||'null');}catch{return null;}}
 function saveDraft(){try{if(text===base)localStorage.removeItem(KEY);else localStorage.setItem(KEY,JSON.stringify({markdown:text,baseHash:hash(base),savedAt:Date.now()}));$('save-status').textContent=text===base?'当前内容与原稿一致。':'草稿已保存在此浏览器。';}catch{$('save-status').textContent='当前浏览器无法保存草稿，请下载修改稿。';} $('draft-badge').hidden=text===base;}
 function idFor(s,index){const num=s.match(/^第\s*(\d+)\s*章/)||s.match(/^(\d+)\.(\d+)/);return num?(num[2]?'section-'+num[1]+'-'+num[2]:'chapter-'+num[1]):'heading-'+index;}
 function safeHref(href){return /^(https?:\/\/|mailto:|#|\.?\.?\/|assets\/)/i.test(href||'')?href:'#';}
 function toHTML(markdown){
  const tokens=[];
  const protectedText=markdown.replace(/\$\$([\s\S]+?)\$\$|(?<!\\)\$([^\n$]+?)(?<!\\)\$/g,(all,display,inline)=>{
   const tex=display||inline;const found=display?catalog.find(f=>normalize(f.wide)===normalize(tex)):null;
   tokens.push({tex:found&&article.clientWidth<650?found.narrow:tex,canonical:tex,display:!!display,number:found?.number});
   return 'RSIEQUATIONTOKEN'+(tokens.length-1)+'END';
  });
  const renderer=new marked.Renderer();
  paragraphEntries=[];SideInnParagraphs.decorate(renderer,{mathTokens:tokens,onParagraph:p=>paragraphEntries.push(p)});
  renderer.html=token=>esc(typeof token==='string'?token:token.text);
  renderer.image=token=>{const [label,...parts]=token.text.split(' | ');return `<figure class="note-figure"><span class="figure-label">${esc(label)}</span><img src="${esc(safeHref(/^\.?\/?assets\//.test(token.href)?'../'+token.href.replace(/^\.?\//,''):token.href))}" alt="${esc(label)}" loading="lazy"><figcaption>${esc(parts.join(' | '))} <a href="${esc(safeHref(/^\.?\/?assets\//.test(token.href)?'../'+token.href.replace(/^\.?\//,''):token.href))}" target="_blank" rel="noopener" aria-label="查看原图：${esc(label)}">查看原图 ↗</a></figcaption></figure>`;};
  let html=marked.parse(protectedText,{gfm:true,renderer});
  html=html.replace(/<p>RSIEQUATIONTOKEN(\d+)END<\/p>/g,(_,n)=>{const t=tokens[n];return t.display?`<div class="equation" data-equation="${t.number||''}"><div class="math-display">\\[${esc(t.tex)}\\]</div></div>`:`<p><span class="math-inline">\\(${esc(t.tex)}\\)</span></p>`;});
  html=html.replace(/RSIEQUATIONTOKEN(\d+)END/g,(_,n)=>{const t=tokens[n];return `<span class="math-inline">\\(${esc(t.tex)}\\)</span>`;});
  return html;
 }
 let typesetQueue=Promise.resolve();
 async function render(markdown){
  const version=++renderVersion;
  await (typesetQueue=typesetQueue.catch(()=>{}).then(async()=>{
   if(version!==renderVersion)return;
   await MathJax.startup.promise;
   MathJax.typesetClear([article]);
   article.innerHTML=toHTML(markdown);article.setAttribute('aria-busy','true');
   for(const a of article.querySelectorAll('a[href]')){a.setAttribute('href',safeHref(a.getAttribute('href')));if(/^https?:/.test(a.getAttribute('href'))){a.target='_blank';a.rel='noopener noreferrer';}}
   for(const table of [...article.querySelectorAll('table')]){const wrap=document.createElement('div');wrap.className='table-scroll';table.before(wrap);wrap.append(table);}
   for(const eq of article.querySelectorAll('.equation')){const p=eq.previousElementSibling;if(p?.tagName==='P'&&p.children.length===1&&p.firstElementChild.tagName==='STRONG'&&/^式\s*\d+/.test(p.textContent)){const caption=document.createElement('div');caption.className='equation-caption';caption.textContent=p.textContent;eq.prepend(caption);p.remove();}}
   headings=[...article.querySelectorAll('h2,h3')];headings.forEach((h,i)=>h.id=idFor(h.textContent,i));
   buildTOC();
   await MathJax.typesetPromise([article]);
   if(version!==renderVersion)return;
   article.setAttribute('aria-busy','false');updateProgress();search();
   const target=location.hash.slice(1);if(target&&!editing)document.getElementById(target)?.scrollIntoView();
  }));
 }
 function buildTOC(){
  $('toc').replaceChildren();let current,chapters=0,subsections=0;
  for(const h of headings){if(h.tagName==='H2'){chapters++;const detail=document.createElement('details');detail.className='toc-chapter';const summary=document.createElement('summary'),link=document.createElement('a');link.href='#'+h.id;link.textContent=h.textContent;summary.append(link);detail.append(summary);const children=document.createElement('div');children.className='toc-subsections';detail.append(children);$('toc').append(detail);current=children;if(chapters===1)detail.open=true;}else if(current){subsections++;const a=document.createElement('a');a.href='#'+h.id;a.textContent=h.textContent;current.append(a);}}
  $('chapter-count').textContent=chapters+' 章';$('note-stats').textContent=chapters+' 章 · '+subsections+' 小节 · '+article.querySelectorAll('.equation').length+' 个公式';
  observer?.disconnect();observer=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting)setActive(e.target);},{rootMargin:'-90px 0px -65% 0px',threshold:0});headings.forEach(h=>observer.observe(h));
 }
 function setActive(h){$('toc').querySelectorAll('.active').forEach(a=>a.classList.remove('active'));$('toc').querySelectorAll('.is-current').forEach(a=>a.classList.remove('is-current'));const link=$('toc').querySelector(`a[href="#${h.id}"]`);if(link){link.classList.add('active');const parent=link.closest('details');parent.open=true;parent.classList.add('is-current');}$('location-label').textContent=h.textContent;}
 function updateProgress(){const available=document.documentElement.scrollHeight-innerHeight;const ratio=available>0?Math.max(0,Math.min(1,scrollY/available)):0;$('reading-progress').style.width=(ratio*100)+'%';$('back-top').hidden=scrollY<600;}
 function clearMarks(){for(const mark of article.querySelectorAll('mark.search-hit'))mark.replaceWith(document.createTextNode(mark.textContent));article.normalize();searchHits=[];hitIndex=-1;}
 function search(){
  clearTimeout(searchTimer);clearMarks();const query=$('search').value.trim();activeQuery=query;$('search-next').hidden=true;$('search-count').textContent='';if(!query)return;
  const walker=document.createTreeWalker(article,NodeFilter.SHOW_TEXT,{acceptNode:node=>node.parentElement.closest('mjx-container,svg,script,style,.equation,.math-inline,.paragraph-action')?NodeFilter.FILTER_REJECT:NodeFilter.FILTER_ACCEPT});const nodes=[];while(walker.nextNode())nodes.push(walker.currentNode);
  const lower=query.toLocaleLowerCase();
  for(const node of nodes){const value=node.textContent,lv=value.toLocaleLowerCase();let start=0,index=lv.indexOf(lower);if(index<0)continue;const fragment=document.createDocumentFragment();while(index>=0){fragment.append(document.createTextNode(value.slice(start,index)));const mark=document.createElement('mark');mark.className='search-hit';mark.textContent=value.slice(index,index+query.length);fragment.append(mark);searchHits.push(mark);start=index+query.length;index=lv.indexOf(lower,start);}fragment.append(document.createTextNode(value.slice(start)));node.replaceWith(fragment);}
  $('search-count').textContent=searchHits.length?searchHits.length+' 处':'无匹配';$('search-next').hidden=!searchHits.length;
 }
 function nextHit(){if(!searchHits.length)return;searchHits[hitIndex]?.classList.remove('current');hitIndex=(hitIndex+1)%searchHits.length;const hit=searchHits[hitIndex];hit.classList.add('current');hit.scrollIntoView({block:'center',behavior:'smooth'});$('search-count').textContent=(hitIndex+1)+' / '+searchHits.length;}
 function setEditing(value){editing=value;document.body.classList.toggle('editing',value);$('editor-panel').hidden=!value;$('edit-toggle').setAttribute('aria-pressed',String(value));$('edit-toggle').textContent=value?'完成编辑':'编辑笔记';if(value){$('editor').value=text;$('editor').focus();}else{saveDraft();}render(text).catch(showError);}
 function menu(open){document.body.classList.toggle('menu-open',open);$('menu-backdrop').hidden=!open;$('mobile-menu').setAttribute('aria-expanded',String(open));}
 function download(){const blob=new Blob([text],{type:'text/markdown;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=document.body.dataset.downloadName||(noteId==='rsi'?'rsi-study-notes.md':noteId+'-notes.md');a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
 $('edit-toggle').addEventListener('click',()=>setEditing(!editing));$('editor-close').addEventListener('click',()=>setEditing(false));
 $('editor').addEventListener('input',()=>{text=$('editor').value;$('save-status').textContent='正在更新预览…';clearTimeout(previewTimer);previewTimer=setTimeout(()=>{saveDraft();render(text).catch(showError);},500);});
 $('restore').addEventListener('click',()=>{text=base;$('editor').value=text;saveDraft();notice('');render(text).catch(showError);});
 $('download').addEventListener('click',download);$('editor-download').addEventListener('click',download);$('print').addEventListener('click',()=>window.print());
 $('mobile-menu').addEventListener('click',()=>menu(!document.body.classList.contains('menu-open')));$('close-menu').addEventListener('click',()=>menu(false));$('menu-backdrop').addEventListener('click',()=>menu(false));
 $('toc').addEventListener('click',e=>{if(e.target.closest('a'))menu(false);});
 $('search').addEventListener('input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(search,120);});$('search').addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();clearTimeout(searchTimer);if(activeQuery!==$('search').value.trim())search();nextHit();}if(e.key==='Escape'){$('search').value='';search();}});$('search-next').addEventListener('click',nextHit);
 $('back-top').addEventListener('click',()=>{location.hash='top';window.scrollTo({top:0,behavior:'smooth'});});addEventListener('scroll',updateProgress,{passive:true});
 let resizeTimer;addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>render(text).catch(showError),250);});
 function showError(error){article.setAttribute('aria-busy','false');notice('笔记暂未完整显示，请刷新页面重试。');console.error(error);}
 async function init(){
  const [response,formulas]=await Promise.all([fetch(source,{cache:'no-store'}),fetch(document.body.dataset.formulaSource||'../assets/formulas.json')]);if(!response.ok||!formulas.ok)throw Error('Unable to load notes');base=await response.text();lastRemote=base;catalog=await formulas.json();const draft=getDraft();text=draft?.baseHash===hash(base)?draft.markdown:base;if(draft&&draft.baseHash!==hash(base))notice('网页原稿已更新。当前显示最新原稿，旧草稿仍保留在此浏览器。');$('draft-badge').hidden=text===base;await render(text);
  if(['localhost','127.0.0.1'].includes(location.hostname))setInterval(async()=>{try{const r=await fetch(source,{cache:'no-store'}),next=await r.text();if(r.ok&&next!==lastRemote){lastRemote=next;if(editing||text!==base)notice('原稿有更新。下载当前草稿后，可刷新页面查看最新原稿。');else{base=text=next;await render(text);}}}catch{}},1500);
 }
 init().catch(showError);
})();
