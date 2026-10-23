import {TOKENS,runFull,runBounded,restorePrefix,decodeOne,positions} from './model.mjs';
const $=id=>document.getElementById(id),full=runFull(),short=runBounded(full,3),decoded=decodeOne(short),restored=restorePrefix(full,3);
const fmt=v=>v.map(x=>(Math.abs(x)<.00005?0:x).toFixed(4)),pair=v=>String.raw`\begin{bmatrix}${fmt(v).join('&')}\end{bmatrix}`;
const frames=[
 ['输入','先把位置固定下来','“请 / 记住 / 暗号 / 是 / 蓝鲸 / 。 / 重复 / 暗号”是人为切成的 8 个位置。以后编号 8 始终是最后那个“暗号”，不要把位置变化和向量变化混为一谈。',6,0,8,'人工 token 切分；不是实际 tokenizer。'],
 ['输入','词变成向量，编号不变','每个 token 查表得到一个二维向量。接下来移动的黄色方块代表这个位置的隐藏状态；文字本身不会被网络逐层改写。',6,0,8,'人工嵌入值；完整矩阵在数值详情中。'],
 ['Encoder','8 个位置一起经过第 1 层','第一层更新所有位置的表示，并准备第一层自己的局部 KV。动画里一起向下移动，表示同层计算；每个位置只能读自己及过去。',6,1,8,'机制：因果约束。6 层排列是教学设定。'],
 ['Encoder','同一批位置进入第 2 层','上一层输出直接成为本层输入。这里仍然计算全部 8 个位置。Encoder 自己的全局缓存 A 不在主画面展开，它不是右侧的 decoder 缓存 C。',6,2,8,'教学编码器结构；A 与 C 分开保存。'],
 ['Encoder','得到完整的第三层输出','到 encoder 末层，8 个位置的表示都已算好。先记住黄色的 8 号方块：下一步的两条路径都从这批第三层表示出发。',7,3,8,'论文 §2.2：decoder 全局 KV 的来源。'],
 ['CED 分叉','第一条路：全部 8 行投影成全局 KV','绿色副本从 8 个位置飞向右侧缓存。它们经过投影后成为 C 的条目，供注意力读取；原来的隐藏状态仍然存在。投影不等于再运行整个 decoder。',8,3,8,'论文 §2.2；attn.py：_store_yoco_full_kv。'],
 ['CED 分叉','第二条路：只有末尾 6、7、8 继续','右侧 C 保留 1–8 的全局内容。主路径只留下末尾 3 个隐藏状态，准备通过 decoder；位置 1–5 的后半层前向被省去。',7,3,8,'论文 §3.2.2：尾部 encoder 输出进入 decoder。'],
 ['Decoder 第 4 层','先从本层输入生成 Q 和局部 KV','6、7、8 各自的第三层表示产生本层主查询 Q 和局部 KV。右下角出现“第 4 层”的局部条目。接下来只放大追踪位置 8；其余保留位置也在计算。',7,3,8,'论文 §2.3：各层独立计算主 Q 与 SWA KV。'],
 ['Decoder 第 4 层','Q 同时读取全局和局部两类内容','绿色流表示从 C 读出全局位置 5、8；蓝色流表示从第 4 层局部缓存读出位置 6、7、8。它们与当前位置的 Q 一起进入注意力计算。缓存没有变成输入 token。',9,3,8,'选址 {5,8} 来自人工模型；数据来源遵循 §2.3。'],
 ['Decoder 第 4 层','注意力计算后，得到新的隐藏状态','位置 8 的状态现在更新为第四层输出。6、7 也有各自的输出。下一层接收这些新状态；右侧 C 不会因为它们变化就重新生成。',7,4,8,'层更新算式是教学简化，数值不是 DeepSeek 激活。'],
 ['Decoder 第 5 层','Reindex：仍用 C，重新选择地址','本层从第四层输出计算自己的 Q、局部 KV 和索引查询。这个例子重新选出全局位置 4、6；内容还是同一份 C，变化的是读取地址。',8,4,8,'论文 §2.3.1：共享内容，重新索引。'],
 ['Decoder 第 5 层','读完之后，状态再次向前','用新的主 Q 读取全局 4、6 和本层局部 6、7、8，得到第五层输出。绿色缓存不动，黄色隐藏状态继续变化。',7,5,8,'真实机制与人工算式的边界见下方核对说明。'],
 ['Decoder 第 6 层','Reuse：地址也沿用，仍计算新输出','本层沿用 C 和上一层的全局地址 4、6，不重新索引。但它仍从第五层状态生成自己的主 Q 和局部 KV，计算注意力。',8,5,8,'论文 §2.3.1：Reuse 不等于跳过整层。'],
 ['Decoder 第 6 层','完成这个 prompt 的 prefill','末尾 3 个位置走完 decoder。现在全局 C 已准备好，decoder 各层的末尾局部状态也已准备好。黄色方块的主路径到这里结束。',7,6,8,'论文 §3.2.2：重放结果用于随后 decode。'],
 ['暂停看状态','省的是哪些位置，保留的是什么？','Encoder 处理了全部 8 个位置；decoder 只处理末尾 3 个。位置 1–5 的全局信息仍在 C 中。局部状态按层保存，不能拿同一份局部 KV 代替所有层。',7,6,8,'层×位置计数不是端到端耗时测量。'],
 ['截断对照','回看第 6 个位置：缺失发生在这里','完整运行时，第 4 层位置 6 的局部窗口是 4、5、6。只重放 6–8 时，4、5 的本层局部 KV 没有生成，于是位置 6 只能读到局部 6。全局 C 没丢，局部读法却变了。',10,4,6,'论文 §3.2.2：重放段外 SWA 不可见。'],
 ['截断对照','第 8 个位置也会间接受影响','本例第 4 层位置 8 的结果仍然相同；但第 6、7 个位置已改变。它们成为第 5 层的局部 KV，所以位置 8 到第 5 层时也改变了。位置范围相同，不代表里面的值相同。',11,5,8,'差异传播是实际计算；误差大小仅属于本教学模型。'],
 ['开始生成','先得到第 9 个 token，再把它送回模型','真实流程是：末尾状态 → 输出头 → 采样 → 第 9 个 token。本演示省略输出头与采样，外部指定“蓝鲸”，只演示这个新输入如何继续运行。',8,0,9,'“蓝鲸”不是该教学网络预测出的答案。'],
 ['Decode','只有一个新位置，但仍经过完整 encoder','9 号方块依次经过前三层。旧位置的局部 KV 直接从缓存读取，不重新前向计算整段 prompt。当前局部窗口从 6、7、8 滑到 7、8、9。',9,3,9,'每层只新增当前位置的 query 和局部 KV。'],
 ['Decode','从新的 encoder 输出追加 C 的第 9 行','第 9 个位置得到第三层表示后，投影产生自己的全局条目。右侧旧的 1–8 行原值保留，只追加第 9 行。',7,3,9,'CED 全局 KV 来源相同，序列长度向前增长。'],
 ['Decode','第 9 个位置继续经过全部 decoder','第 4 层、5 层、6 层都要运行。每层用当前 Q 读取选中的全局条目，以及本层局部 7、8、9。这里没有“生成阶段也跳过后半模型”。',10,6,9,'本例 decode 的第 5 层在候选池内重新索引。'],
 ['Decode','得到下一次预测所需的末尾状态','第 9 个位置走完整个模型，得到用于预测第 10 个 token 的状态。全局缓存增长到 9 行；各层当前的局部可见位置是 7、8、9。',7,6,9,'逻辑窗口与物理环形缓冲区槽位不是同一个概念。'],
 ['另一请求','稍后：原来的 8-token 前缀命中','切换到另一次请求，它复用原来的位置 1–8。这里特意假设全局 A、C 仍在，但局部 SWA 状态已经缺失。如果局部也命中，就不需要这段 encoder 重放。',9,0,8,'论文 §3.2.1–3.2.2：全局命中、局部缺失的分支。'],
 ['恢复局部状态','重新送入原前缀的末尾 3 个 token','从位置 6、7、8 的原始嵌入开始，经过 encoder 恢复局部状态。命中的旧全局缓存只读取、不覆盖；近似恢复的第三层表示可能与原来不同。',10,3,8,'官方 Q&A：replayed prefix 的 global KV 原样复用。'],
 ['恢复局部状态','再准备 decoder 的局部状态','恢复出的尾部表示继续走过 decoder。右侧旧 C 仍然不变；新的近似局部状态供当前请求使用，不作为 decoder 的永久前缀状态保存。',10,6,8,'论文 §3.2.2：decoder SWA 用于 decode，不写前缀缓存。'],
 ['回到完整流程','现在分清这三样东西','黄色：逐层传递的隐藏状态。绿色：可跨层共享的全局内容。蓝色：本层生成的局部状态。CED 提前准备绿色缓存，有限重放只为短尾部恢复蓝色状态；黄色主路径仍要穿过保留的各层。',10,6,8,'可拖动进度回看；详细算式单独保留在下方入口。']
];
const Y=[65,127,189,251,353,415,477],X=i=>137+(i-1)*68,C=i=>[761+((i-1)%3)*69,219+Math.floor((i-1)/3)*49];
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const text=(x,y,s,size=15,cls='',anchor='start')=>`<text x="${x}" y="${y}" font-size="${size}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`;
const rect=(x,y,w,h,fill,stroke='none',rx=7)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}"/>`;
const path=(d,cls='wire',end=true)=>`<path d="${d}" class="${cls}"${end?' marker-end="url(#arrow)"':''}/>`;
const lerp=(a,b,t)=>a+(b-a)*t,ease=t=>t<.5?2*t*t:1-(-2*t+2)**2/2;
const loc=(l,j)=>[822+(j-6)*43,431+(l-1)*25];
let time=0,playing=false,last=0,raf=0,frame=-1,ready=false,speed=1,reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
const duration=frames.reduce((s,f)=>s+f[3],0),starts=frames.map((_,i)=>frames.slice(0,i).reduce((s,f)=>s+f[3],0));
function at(t){let i=frames.length-1;while(i>0&&t<starts[i])i--;return [i,Math.min(1,(t-starts[i])/frames[i][3])];}
const mathCache=new Map();
async function tex(s){if(mathCache.has(s))return mathCache.get(s);const n=await MathJax.tex2svgPromise(s,{display:false});if(n.querySelector('[data-mml-node="merror"]'))throw Error('Math formula failed');n.querySelectorAll('mjx-assistive-mml').forEach(e=>e.remove());n.setAttribute('aria-label',s);const out=n.outerHTML;mathCache.set(s,out);return out;}
const valFormula=(l,i,s=short)=>String.raw`h_{${i}}^{(${l})}=${pair(s.h[l][i])}`;
let values=[];
async function prepare(){
 for(let i=0;i<frames.length;i++){const f=frames[i],s=i>=23?restored:i>=17&&i<=21?decoded:short;let v=valFormula(f[4],f[5],s);
 if(i===5)v=String.raw`c_8=h_8^{(3)}P_C=${pair(short.shared.C[8])}`;
 if(i===7||i===8)v=String.raw`q_8^{(4)}=${pair(short.traces[4][8].q)}`;
 if(i===10)v=String.raw`q_8^{(5)}=${pair(short.traces[5][8].q)}`;
 if(i===12)v=String.raw`q_8^{(6)}=${pair(short.traces[6][8].q)}`;
 if(i===15)v=String.raw`h_6^{(4)}:\quad\text{全 }${pair(full.h[4][6])}\quad\text{重 }${pair(short.h[4][6])}`;
 if(i===16)v=String.raw`h_8^{(5)}:\quad\text{全 }${pair(full.h[5][8])}\quad\text{重 }${pair(short.h[5][8])}`;
 if(i===19)v=String.raw`c_9=h_9^{(3)}P_C=${pair(decoded.shared.C[9])}`;
 if(i===22)v=String.raw`C_{1:8}\text{ 命中},\quad\mathrm{KV}_{\mathrm{SWA}}\text{ 缺失}`;
 values[i]=await tex(v);
 }
}
function token(i,x,y,label,fill='#f1d6a0',opacity=1){return `<g opacity="${opacity}">${rect(x-25,y-19,50,38,fill,i===8?'#9e6c20':'none')}${text(x,y-1,String(i),17,'','middle')}${text(x,y+13,label,10,'','middle')}</g>`;}
function packet(x1,y1,x2,y2,p,fill='#287864',label=''){const t=ease(Math.min(1,Math.max(0,p))),x=lerp(x1,x2,t),y=lerp(y1,y2,t);return `<g>${rect(x-16,y-12,32,24,fill)}${label?`<text x="${x}" y="${y+4}" style="fill:white" font-size="11" text-anchor="middle">${label}</text>`:''}</g>`;}
function background(i){let s=`<title id="board-title">${esc(frames[i][1])}</title><desc id="board-desc">${esc(frames[i][2])}</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#81998b"/></marker></defs>`;
 s+=rect(16,22,696,262,'#f3f7ef')+rect(16,307,696,204,'#fcf6eb');s+=text(30,43,'ENCODER · 前 3 层',14)+text(30,328,'DECODER · 后 3 层',14);
 for(let l=0;l<=6;l++){s+=text(34,Y[l]+5,l===0?'输入':`第 ${l} 层`,14);if(l>3)s+=text(34,Y[l]+22,['Full','Reindex','Reuse'][l-4],10,'sub');for(let j=1;j<=8;j++){s+=rect(X(j)-26,Y[l]-20,52,40,'none','#dce5d9',5);if(l>3&&j<6&&i>=6)s+=text(X(j),Y[l]+4,'省去',11,'sub','middle');}}
 s+=text(45,296,'同一列 = 同一个位置；向下 = 经过下一层',13,'sub');
 s+=rect(728,155,255,222,'#eaf4ee','#bbd3c5');s+=text(745,179,'全局 C：decoder 共享',16)+text(745,198,'由完整 encoder 输出投影',12,'sub');
 s+=rect(728,385,255,183,'#f0f5f7','#c5d3d7');s+=text(745,403,'各层自己的局部 KV',14);s+=text(741,590,'Encoder 全局 A 未展开；与 C 分开。',11,'sub');
 return s;
}
function localLevel(i,p){if(i<2)return 0;if(i<=4)return i-1;if(i<=6)return 3;if(i<=9)return 4;if(i<=11)return 5;if(i<22)return 6;if(i===22)return 0;if(i===23)return Math.min(3,Math.floor(p*3)+1);return i===24?3+Math.min(3,Math.floor(p*3)+1):6;}
function draw(i,p){let s=background(i),progress=reduced?1:ease(Math.min(1,p/.55)),localDone=localLevel(i,p),numC=i<5?0:(i>=19&&i<=21?9:8);
 for(let j=1;j<=9;j++){const [x,y]=C(j);let visible=j<=numC&&(i!==5||p>.55)&&!(j===9&&i===19&&p<.55);let selected=[];
 if([8,9].includes(i))selected=short.traces[4][8].selected;if(i>=10&&i<=14)selected=short.traces[i<12?5:6][8].selected;if(i===20)selected=decoded.traces[Math.min(6,4+Math.floor(p*3))][9].selected;
 s+=rect(x-25,y-15,50,32,visible?(selected.includes(j)?'#9ccdb3':'#d8eadd'):'#f8fbf8',selected.includes(j)?'#287864':'#cadbd0',4)+text(x,y+5,visible?`C · ${j}`:j===9?'待追加':'空',12,visible?'':'sub','middle');}
 for(let l=1;l<=6;l++){s+=text(742,loc(l,6)[1]+4,`层 ${l}`,11,'sub');for(let j=6;j<=8;j++){const [x,y]=loc(l,j),changed=i===16&&l===5&&j<8,show=l<=localDone;const written=i===18?l<=Math.min(3,Math.floor(p*3)+1):i===19?l<=3:i===20?l<=Math.min(6,4+Math.floor(p*3)):i===21;const slot=written?j+1:j;s+=rect(x-16,y-9,32,20,show?(changed?'#e8b0a3':'#c8dce1'):'#fff',show?'none':'#d4dfe1',3)+text(x,y+5,show?String(slot):'—',11,'','middle');}}
 // Fixed historical outputs stay on the board; moving tokens are the active computation.
 if(i>=5&&i<=21){for(let j=1;j<=8;j++)s+=token(j,X(j),Y[3],'编码后','#e5d4b1',.25);}
 let range=positions(1,8),y=Y[0],label='token',fill='#f1d6a0';
 if(i===1)label='嵌入';
 if(i>=2&&i<=4){const l=i-1;y=lerp(Y[l-1],Y[l],progress);label=progress<.8?'计算中':`层${l}输出`;}
 if(i===5){y=Y[3];label='层3输出';for(let j=1;j<=8;j++){const [cx,cy]=C(j);s+=path(`M${X(j)} ${Y[3]} Q710 120 ${cx} ${cy}`,'global-wire',false);s+=packet(X(j),Y[3],cx,cy,progress,'#287864',String(j));}}
 if(i===6){y=Y[3];label='继续';for(let j=1;j<=5;j++)s+=token(j,X(j),y,'到此停止','#e5e8df',1-progress);range=[6,7,8];}
 if(i>=7&&i<=14){range=[6,7,8];let l=i<=9?4:i<=11?5:6;const output=[9,11,13,14].includes(i);y=output?lerp(Y[l-1],Y[l],progress):Y[l-1];label=output?`层${l}输出`:'本层输入';
 if(i===14){y=Y[6];label='已完成';}
 if([7,8,10,12].includes(i)){s+=rect(98,519,608,83,'#f8ecd5','#ddc6a0');s+=text(116,541,`第 ${l} 层注意力：主 Q + 局部 KV + 全局 KV → 更新状态`,14);s+=token(8,160,574,'本层 Q');
 const selected=short.traces[l][8].selected;
 s+=path(`M${X(8)} ${Y[l-1]+20} L${X(8)} 515 L160 515 L160 552`);
 if(i!==7){for(const j of selected){const [cx,cy]=C(j),endX=485+selected.indexOf(j)*60;s+=path(`M${cx} ${cy} Q718 510 ${endX} 574`,'global-wire',false);s+=packet(cx,cy,endX,574,progress,'#287864',`C${j}`);}for(let j=6;j<=8;j++){const [lx,ly]=loc(l,j);s+=packet(lx,ly,270+(j-6)*52,574,progress,'#789ca6',String(j));}}
 else{for(let j=6;j<=8;j++){const [lx,ly]=loc(l,j);s+=packet(X(j),Y[l-1],lx,ly,progress,'#789ca6',String(j));}}
 }
 }
 if(i===15||i===16){range=[6,7,8];y=Y[i===15?4:5];label='重放结果';for(const j of [6,7])s+=token(j,X(j),Y[4],'已改变','#e8ad9f');
 if(i===15){for(const j of [4,5]){s+=rect(X(j)-26,Y[4]-20,52,40,'#f9e8e3','#bd6b57',5);s+=text(X(j),Y[4]+5,'缺失',12,'','middle');s+=path(`M${X(j)} ${Y[4]+22} Q${X(j)} 400 ${X(6)} ${Y[4]+20}`,'error-wire');}s+=text(118,553,'位置 6 的局部读取：完整 {4,5,6} → 重放 {6}',20);}
 else{for(const j of [6,7]){s+=path(`M${X(j)} ${Y[4]+20} L${X(8)} ${Y[5]-20}`,'error-wire');s+=packet(X(j),Y[4],X(8),Y[5],progress,'#b64b39',String(j));}s+=text(118,553,'下层状态改变 → 上层局部 KV 改变 → 位置 8 的结果改变',17);}
 }
 if(i>=17&&i<=21){range=[];for(let j=6;j<=8;j++)s+=token(j,X(j),Y[6],'旧状态','#e6d7b9',.45);
 let newY=Y[0],newLabel='新 token';
 if(i===17){s+=rect(125,524,550,64,'#f8ecd5');s+=text(145,547,'末尾状态 → 输出头 / 采样（本例省略）',16);s+=text(145,574,'外部指定新 token：“蓝鲸”，位置编号 9',16);s+=path(`M${X(8)} ${Y[6]} Q720 520 ${X(9)} ${Y[0]}`);newY=lerp(Y[6],Y[0],progress);}
 if(i===18){const t=Math.min(2.999,p*3),l=Math.floor(t)+1;newY=lerp(Y[l-1],Y[l],reduced?1:ease(t%1));newLabel=`层${l}计算`;}
 if(i===19){newY=Y[3];newLabel='层3输出';const [cx,cy]=C(9);s+=path(`M${X(9)} ${Y[3]} L${cx} ${cy}`,'global-wire');s+=packet(X(9),Y[3],cx,cy,progress,'#287864','C9');}
 if(i===20){const t=Math.min(2.999,p*3),l=4+Math.floor(t);newY=lerp(Y[l-1],Y[l],reduced?1:ease(t%1));newLabel=`层${l}计算`;const selected=decoded.traces[l][9].selected;for(const j of selected){const [cx,cy]=C(j);s+=packet(cx,cy,X(9),newY,ease(t%1),'#287864',`C${j}`);}s+=text(120,550,`第 ${l} 层：局部 {7,8,9}；全局 {${selected.join(',')}}`,17);}
 if(i===21){newY=Y[6];newLabel='层6输出';s+=text(120,550,'这个末尾状态将用于下一次预测第 10 个 token。',18);}
 s+=token(9,X(9),newY,newLabel,'#f1d6a0');s+=text(X(9),35,'新位置',12,'','middle');
 }
 if(i>=22){range=[6,7,8];let l=0;y=Y[0];label='旧尾部';if(i===23||i===24){let t=Math.min(2.999,p*3);l=(i===23?1:4)+Math.floor(t);y=lerp(Y[l-1],Y[l],reduced?1:ease(t%1));label=`恢复层${l}`;}if(i===25){y=Y[6];label='已恢复';}
 s+=text(118,550,i===22?'换到另一请求：仍以原来的位置 1–8 为前缀':'旧全局 C 保持原值；只恢复当前请求所需的局部状态。',17);
 }
 for(const j of range)s+=token(j,X(j),y,i===0?TOKENS[j-1]:label,((i===15&&j<8)||i===16)?'#e8ad9f':fill);
 if(i===0||i===1)s+=text(118,553,'先认编号：后面追踪的始终是同一个位置，不是换了一批 token。',17);
 $('board').innerHTML=s;
}
function status(i){if(i<5)return '全局 C 尚未构造。encoder 的局部状态逐层建立。';if(i<17)return 'C：位置 1–8 保留。局部 KV：按层独立保存。';if(i<19)return '读取旧缓存，开始为新位置生成状态。';if(i<22)return 'C：追加至位置 9。局部窗口：7、8、9。';return '另一请求：旧 C 的位置 1–8 不覆盖，局部状态重新恢复。';}
function update(t,force=false){time=Math.min(duration,Math.max(0,t));const [i,p]=at(time);
 if(i!==frame||force){frame=i;const f=frames[i];$('phase').textContent=f[0];$('frame-title').textContent=f[1];$('narration').textContent=f[2];$('math-value').innerHTML=values[i];$('tracking').textContent=`位置 ${f[5]} · ${TOKENS[f[5]-1]}`;$('cache-status').textContent=status(i);$('source-label').textContent=f[6];$('frame-count').textContent=`第 ${i+1} / ${frames.length} 页`;$('prev').disabled=i===0;$('next').disabled=i===frames.length-1;
 const group=i<5?0:i<15?5:i<17?15:i<22?17:22;document.querySelectorAll('[data-jump]').forEach(b=>b.setAttribute('aria-current',String(Number(b.dataset.jump)===group)));
 }
 $('timeline').value=Math.min(26,i+p);$('timeline').setAttribute('aria-valuetext',`第 ${i+1} 幕：${frames[i][1]}`);draw(i,p);
}
function stop(){playing=false;cancelAnimationFrame(raf);$('play').textContent=time>=duration?'重新放映':'自动放映';$('play').setAttribute('aria-pressed','false');}
function tick(now){if(!playing)return;if(last)update(time+Math.min((now-last)/1000,.2)*speed);last=now;if(time>=duration){stop();return;}raf=requestAnimationFrame(tick);}
function play(){if(!ready)return;cancelAnimationFrame(raf);if(time>=duration)update(0);playing=true;last=0;$('play').textContent='暂停放映';$('play').setAttribute('aria-pressed','true');raf=requestAnimationFrame(tick);}
function jump(i){stop();const target=Math.max(0,Math.min(frames.length-1,i));update(starts[target],true);if(reduced){update(starts[target]+frames[target][3]*.99);return;}let began;function pageTurn(now){began??=now;const p=Math.min(1,(now-began)/2200);update(starts[target]+frames[target][3]*.99*p);if(p<1)raf=requestAnimationFrame(pageTurn);}raf=requestAnimationFrame(pageTurn);}
async function init(){await MathJax.startup.promise;await prepare();ready=true;
 const hash=location.hash.slice(1),aliases={input:0,encoder:2,fork:5,full:7,reindex:10,reuse:12,replay:15,decode:17,pool:20,restore:22,summary:25};
 update(starts[aliases[hash]??0],true);$('play').setAttribute('aria-pressed','false');
 $('play').addEventListener('click',()=>playing?stop():play());$('restart').addEventListener('click',()=>{stop();update(0,true);play();});$('prev').addEventListener('click',()=>jump(frame-1));$('next').addEventListener('click',()=>jump(frame+1));
 $('speed').addEventListener('change',()=>{speed=Number($('speed').value);});$('timeline').addEventListener('input',()=>{stop();const v=Number($('timeline').value),i=Math.min(25,Math.floor(v));update(v>=26?duration:starts[i]+(v-i)*frames[i][3],true);});
 document.querySelectorAll('[data-jump]').forEach(b=>b.addEventListener('click',()=>{jump(Number(b.dataset.jump));}));
 $('zoom').addEventListener('click',()=>{const on=$('screen').classList.toggle('zoomed');$('zoom').setAttribute('aria-pressed',String(on));$('zoom').textContent=on?'适应宽度':'放大画面';});
 document.addEventListener('keydown',e=>{if(e.target.closest('input,select,textarea,a'))return;if(e.key==='ArrowRight'){e.preventDefault();jump(frame+1);}if(e.key==='ArrowLeft'){e.preventDefault();jump(frame-1);}});
 document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
 if(reduced)$('motion-note').textContent='已尊重“减少动态效果”设置，可逐幕查看。';
 $('board').dataset.ready='true';
}
init().catch(e=>{$('error').hidden=false;console.error(e);});
