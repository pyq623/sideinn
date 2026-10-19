/* Stable paragraph identity shared by the reader and the build-time registry. */
(function(root){
 'use strict';
 const normalize=s=>String(s).normalize('NFKC').replace(/\s+/g,' ').trim();
 const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 function hash(s){let a=2166136261,b=2246822507;for(let i=0;i<s.length;i++){a=Math.imul(a^s.charCodeAt(i),16777619);b=Math.imul(b^s.charCodeAt(i),3266489909);}return (a>>>0).toString(16).padStart(8,'0')+(b>>>0).toString(16).padStart(8,'0');}
 function protect(markdown){const tokens=[];const text=markdown.replace(/\$\$([\s\S]+?)\$\$|(?<!\\)\$([^\n$]+?)(?<!\\)\$/g,(all,display,inline)=>{tokens.push({canonical:display||inline,display:!!display});return 'RSIEQUATIONTOKEN'+(tokens.length-1)+'END';});return {text,tokens};}
 function decorate(renderer,{mathTokens=[],onParagraph=()=>{}}={}){
  let section='intro',sectionTitle='笔记导读',ordinal=0;const seen=new Map();
  const restore=s=>String(s||'').replace(/RSIEQUATIONTOKEN(\d+)END/g,(_,i)=>{const t=mathTokens[+i];return t?(t.display?'$$':'$')+(t.canonical??t.tex)+(t.display?'$$':'$'):'';});
  function entry(raw){const canonical=normalize(restore(raw));const key=section+'-'+hash(canonical);const occurrence=(seen.get(key)||0)+1;seen.set(key,occurrence);const id=key+(occurrence>1?'-'+occurrence:'');
   const text=restore(raw).replace(/!\[[^\]]*\]\([^)]*\)/g,'').replace(/\[([^\]]+)\]\([^)]*\)/g,'$1').replace(/\[([^\]]+)\]\[[^\]]*\]/g,'$1').replace(/\*\*|__|`/g,'').replace(/^\s*[-*+]\s+/gm,'').replace(/\s+/g,' ').trim();
   const value={id,text,section,sectionTitle,ordinal:++ordinal};onParagraph(value);return ` data-paragraph-id="${escape(id)}" data-paragraph-number="${ordinal}"`;}
  const heading=renderer.heading,paragraph=renderer.paragraph,listitem=renderer.listitem;
  renderer.heading=function(token){if(token.depth===2||token.depth===3){const t=token.text.replace(/\*|`/g,'');const n=t.match(/^第\s*(\d+)\s*章/)||t.match(/^(\d+)\.(\d+)/);section=n?(n[2]?'section-'+n[1]+'-'+n[2]:'chapter-'+n[1]):'heading-'+hash(t);sectionTitle=t;}return heading.call(this,token);};
  renderer.paragraph=function(token){
   const content=token.text.trim(),m=content.match(/^RSIEQUATIONTOKEN(\d+)END$/);
   if(m&&mathTokens[+m[1]]?.display)return paragraph.call(this,token);
   if(token.tokens?.length===1&&token.tokens[0].type==='strong'&&/^式\s*\d+/.test(token.tokens[0].text))return paragraph.call(this,token);
   if(token.tokens?.length&&token.tokens.every(t=>t.type==='image'||(t.type==='text'&&!t.text.trim())))return this.parser.parseInline(token.tokens)+'\n';
   return '<p'+entry(token.raw||token.text)+'>'+this.parser.parseInline(token.tokens)+'</p>\n';
  };
  renderer.listitem=function(token){const html=listitem.call(this,token);if(/data-paragraph-id=/.test(html)||token.tokens?.some(t=>t.type==='list'))return html;return html.replace('<li>','<li'+entry(token.text||token.raw)+'>');};
  return renderer;
 }
 root.SideInnParagraphs={normalize,hash,protect,decorate};
})(typeof globalThis!=='undefined'?globalThis:this);
