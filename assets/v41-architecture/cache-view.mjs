// A teaching view of the four cache sources, not a physical allocation timeline.
export const cacheBanks=[
 {name:'Encoder · 第 1 套',source:'层 2 构造 · 压缩率 2',users:'层 2–7 使用'},
 {name:'Encoder · 第 2 套',source:'层 8 构造 · 压缩率 2',users:'层 8–13 使用'},
 {name:'Encoder · 第 3 套',source:'层 14 构造 · 压缩率 2',users:'层 14–19 使用'},
 {name:'Decoder · 第 1 套',source:'Encoder 最终表示经 CED 投影 · 压缩率 1',users:'层 20–39 共享'}
];
export function cacheView(step){
 const ready=cacheBanks.map((_,i)=>i<3?i<step.cache[0]:step.cache[1]===1);
 let active=[],badge='',flow=['隐藏状态是原材料','main KV 与 indexer K 合称 global KV','从 Encoder 层 2 开始构造'],note='沿层前向时，Full 建立新的一套；Reuse 沿用已有的一套。';
 const set=(banks,label,items,explanation)=>{active=banks;badge=label;flow=items;note=explanation;};
 switch(step.id){
  case 'map':case 'input':case 'swa':case 'engram':
   note='最初两层只有局部 SWA KV；Engram 是模型的条件记忆。两者都不计入这里的 global KV。';break;
  case 'encoder-full':
   set([0],'构造并读取',['层 2 的隐藏状态','两位置合成 main KV，再投影 indexer K','检索选址 → 读取本套内容'],'左侧三套分别由不同 Full 层构造；不是把各层输出塞进同一个大缓存。');break;
  case 'encoder-reuse':
   set([0],'复用内容与地址',['层 2 的 main KV + indexer K','层 3–7 沿用已有 Top-K 地址','各层 main Q + SWA → 新输出'],'这五层不再构造 global KV，也不运行 indexer；隐藏状态和注意力输出仍逐层更新。');break;
  case 'encoder-group2':case 'encoder-group3':{
   const bank=step.id==='encoder-group2'?1:2,layer=bank===1?8:14;
   set([bank],'新建 → 后五层复用',[`层 ${layer} 的隐藏状态`,`建立 Encoder 第 ${bank+1} 套`,`层 ${layer+1}–${layer+5} 复用这套及其地址`],'下一组有自己的缓存来源，不覆盖前一组；图中的同一个 Full 方框代表不同的实际层。');break;
  }
  case 'hidden':
   flow=['Encoder 最后一层输出','仍是隐藏状态','下一步经 CED 构造 Decoder 缓存'];
   note='三套 Encoder 缓存与 Encoder 最终隐藏状态是不同的数据；Decoder 使用后者来构造自己的缓存。';break;
  case 'global-path':
   set([3],'CED 构造',['Encoder 全部位置的最终隐藏状态','投影为 main KV → 投影 indexer K','形成 Decoder 的一套 global KV'],'这是 Decoder 新的一套，包含 main KV 和 indexer K；不复制 Encoder 的三套缓存。压缩率 1 不合并相邻位置。');break;
  case 'tail-path':
   set([3],'已构造 · 保持共享',['尾部隐藏状态继续前向','逐层准备局部 SWA KV','global KV 仍用 CED 的同一套'],'SWA KV 是另一类局部缓存，不装进下面这些 global KV 框；这一步不新建另一套全局内容。');break;
  case 'decoder-full':
   set([3],'检索并读取',['本层 indexer Q → 共享 indexer K','评分 → Top-512 地址 → 取 main KV','本层 main Q + 取回内容 + SWA → 新输出'],'两种 Q 都来自 Decoder 本层隐藏状态：indexer Q 负责选址，main Q 负责注意力读取。第 10 步构造、第 12 步读取的是同一套。');break;
  case 'candidate':
   set([3],'选址 · 建立候选池',['同一轮 indexer 评分','选 Top-512 positions；另选 Top-2048 blocks','每块 8 个位置 → 最多 16384 个候选地址'],'候选池和 Top-K 是地址信息，不是又一套 main KV；实际内容仍在 Decoder 的这一套缓存中。');break;
  case 'decoder-reuse0':case 'reindex-reuse':
   set([3],'复用内容与地址',['Decoder 同一套 global KV','沿用最近 Full / Reindex 的 Top-K','各层 main Q + SWA → 新输出'],'global KV 的内容来源不变；Reuse 不计算 indexer Q、不重新评分，仍计算自己的 main Q。');break;
  case 'reindex':case 'decoder-group2':case 'decoder-group3':case 'decoder-group4':
   set([3],'缓存不变 · 重选地址',['本层 indexer Q → 共享 indexer K','在候选池内评分 → 新 Top-512','读取同套 main KV；后续三层沿用地址'],'Reindex 改的是选中哪些位置，不重建 main KV 或 indexer K；后续 Reuse 继承新的地址。');break;
  case 'decode':
   set([0,1,2,3],'随新位置更新',['新位置经过全部 40 层','Encoder：完整相邻对形成新条目','Decoder：新位置形成新条目'],'仍是三套 Encoder 来源加一套 Decoder 来源。随序列增长增加条目，不是随层数增加缓存套数；各层仍按既定规则共享。');break;
  case 'dspark':
   flow=['DSpark 提出草稿','目标模型验证','接受后继续生成'];
   note='这里仅展示主干的四套 global KV 来源；草稿验证的临时状态与接受/回退流程见 DSpark 讲解。';break;
  case 'recap':
   flow=['Full：构造 main KV + indexer K','Reindex：内容共享，重新选址','Reuse：内容与地址都复用'];
   note='沿层变化的是缓存来源和选址方式；沿序列增加的是缓存条目。main Q 与局部 SWA KV 仍由各层计算。';break;
 }
 return {ready,active,badge,flow,note};
}
export function mountCacheView(root){
 const cards=cacheBanks.map(bank=>{
  const card=document.createElement('article');card.className='kv-bank';
  const title=document.createElement('h4');title.textContent=bank.name;
  const status=document.createElement('span');status.className='kv-bank-status';
  const parts=document.createElement('div');parts.className='kv-parts';
  for(const [name,meaning] of [['main KV','内容'],['indexer K','检索键']]){
   const part=document.createElement('div'),strong=document.createElement('strong'),small=document.createElement('span');
   strong.textContent=name;small.textContent=meaning;part.append(strong,small);parts.append(part);
  }
  const source=document.createElement('p');source.className='kv-source';source.textContent=bank.source;
  const users=document.createElement('p');users.className='kv-users';users.textContent=bank.users;
  card.append(title,status,parts,source,users);root.querySelector('.kv-banks').append(card);return {card,status};
 });
 return step=>{
  const state=cacheView(step);
  cards.forEach(({card,status},i)=>{
   const active=state.active.includes(i);card.classList.toggle('is-ready',state.ready[i]);card.classList.toggle('is-current',active);
   status.textContent=active?state.badge:state.ready[i]?'已讲到此来源':'尚未构造';
  });
  root.querySelector('.kv-flow').replaceChildren(...state.flow.map((item,i)=>{
   const el=document.createElement('span');el.className='kv-flow-item';
   const num=document.createElement('b');num.textContent=String(i+1);el.append(num,document.createTextNode(item));return el;
  }));
  root.querySelector('.kv-reading').textContent=state.note;
 };
}
