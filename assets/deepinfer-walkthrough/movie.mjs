import {TOKENS,runFull,runBounded,restorePrefix,decodeOne,positions} from './model.mjs';
const $=id=>document.getElementById(id),full=runFull(),short=runBounded(full,3),decoded=decodeOne(short),restored=restorePrefix(full,3);
const fmt=v=>v.map(x=>(Math.abs(x)<.00005?0:x).toFixed(4)),pair=v=>String.raw`\begin{bmatrix}${fmt(v).join('&')}\end{bmatrix}`;
const frames=[
 ['输入','先把位置固定下来','“请 / 记住 / 暗号 / 是 / 蓝鲸 / 。 / 重复 / 暗号”是人为切成的 8 个位置。以后编号 8 始终是最后那个“暗号”，不要把位置变化和向量变化混为一谈。',6,0,8,'人工 token 切分；不是实际 tokenizer。'],
 ['输入','词变成向量，编号不变','每个 token 查表得到一个二维向量。接下来移动的黄色方块代表这个位置的隐藏状态；文字本身不会被网络逐层改写。',6,0,8,'人工嵌入值；完整矩阵在数值详情中。'],
 ['Encoder','8 个位置一起经过第 1 层','第一层更新所有位置的表示，并准备第一层自己的局部 KV。动画里一起向下移动，表示同层计算；每个位置只能读自己及过去。',6,1,8,'机制：因果约束。6 层排列是教学设定。'],
 ['Encoder','同一批位置进入第 2 层','上一层输出直接成为本层输入。这里仍然计算全部 8 个位置。Encoder 自己的全局缓存 A 不在主画面展开，它不是右侧的 decoder 缓存 C。',6,2,8,'教学编码器结构；A 与 C 分开保存。'],
 ['Encoder','得到完整的第三层输出','到 encoder 末层，8 个位置的表示都已算好。先记住黄色的 8 号方块：下一步的两条路径都从这批第三层表示出发。',7,3,8,'论文 §2.2：decoder 全局 KV 的来源。'],
 ['CED 分叉','第一条路：全部 8 行投影成全局 KV','接下来，encoder 的输出会分成两路：一路把全部 8 个位置的表示投影成全局 KV，供 decoder 读取历史；另一路让末尾 3 个位置的隐藏状态继续经过 decoder，计算输出并准备局部 KV。绿色副本从 8 个位置飞向右侧缓存。它们经过投影后成为 C 的条目，供注意力读取；原来的隐藏状态仍然存在。投影不等于再运行整个 decoder。',8,3,8,'论文 §2.2；attn.py：_store_yoco_full_kv。'],
 ['CED 分叉','第二条路：只有末尾 6、7、8 继续','右侧 C 保留 1–8 的全局内容。主路径只留下末尾 3 个隐藏状态，准备通过 decoder；位置 1–5 的后半层前向被省去。',7,3,8,'论文 §3.2.2：尾部 encoder 输出进入 decoder。'],
 ['Decoder 第 4 层','先从本层输入生成 Q 和局部 KV','6、7、8 各自的第三层表示产生本层主查询 Q 和局部 KV。右下角出现“第 4 层”的局部条目。接下来只放大追踪位置 8；其余保留位置也在计算。',7,3,8,'论文 §2.3：各层独立计算主 Q 与 SWA KV。'],
 ['Decoder 第 4 层','先用 Top-k 选位置，再用主 Q 读取内容','全局位置 5、8 是怎么来的？索引器先从位置 8 的本层输入生成自己的索引查询，为可见的全局位置评分；Top-k index selection 就是选出得分最高的 k 个位置索引，本例取 2 个，得到 5、8。随后按这些索引从 C 读取 KV，与窗口内的局部 KV（6、7、8）一起交给主注意力 Q 计算。局部位置由窗口确定，不参加这次 Top-k 筛选。',14,3,8,'论文 §2.3：索引选址与主注意力分工；Top-2 和评分是教学设定。'],
 ['Decoder 第 4 层','注意力计算后，得到新的隐藏状态','位置 8 的状态现在更新为第四层输出。6、7 也有各自的输出。下一层接收这些新状态；右侧 C 不会因为它们变化就重新生成。',7,4,8,'层更新算式是教学简化，数值不是 DeepSeek 激活。'],
 ['Decoder 第 5 层','Reindex：仍用 C，重新选择地址','本层从第四层输出计算自己的主 Q、局部 KV 和索引查询。索引器重新评分并做 Top-2 选择，本例得到全局位置 4、6；内容还是同一份 C，变化的是读取地址。',8,4,8,'论文 §2.3.1：共享内容，重新索引。'],
 ['Decoder 第 5 层','读完之后，状态再次向前','用新的主 Q 读取全局 4、6 和本层局部 6、7、8，得到第五层输出。绿色缓存不动，黄色隐藏状态继续变化。',7,5,8,'真实机制与人工算式的边界见下方核对说明。'],
 ['Decoder 第 6 层','Reuse：地址也沿用，仍计算新输出','本层沿用 C 和上一层选出的 Top-2 位置索引 4、6，不重新评分或执行 Top-k 选择。但它仍从第五层状态生成自己的主 Q 和局部 KV，计算注意力。',8,5,8,'论文 §2.3.1：Reuse 不等于跳过整层。'],
 ['Decoder 第 6 层','完成这个 prompt 的 prefill','末尾 3 个位置走完 decoder。现在全局 C 已准备好，decoder 各层的末尾局部状态也已准备好。黄色方块的主路径到这里结束。',7,6,8,'论文 §3.2.2：重放结果用于随后 decode。'],
 ['开始生成','接着刚才的 prefill，送入第 9 个 token','接下来，使用末尾 3 个位置经过 decoder 后的状态和缓存继续生成。真实流程是：末尾状态 → 输出头 → 采样 → 第 9 个 token。本演示省略输出头与采样，外部指定“蓝鲸”，接着展示这个新输入如何运行。',11,0,9,'“蓝鲸”不是该教学网络预测出的答案。'],
 ['Decode','只有一个新位置，但仍经过完整 encoder','9 号方块依次经过前三层。旧位置的局部 KV 直接从缓存读取，不重新前向计算整段 prompt。当前局部窗口从 6、7、8 滑到 7、8、9。',9,3,9,'每层只新增当前位置的 query 和局部 KV。'],
 ['Decode','从新的 encoder 输出追加 C 的第 9 行','第 9 个位置得到第三层表示后，投影产生自己的全局条目。右侧旧的 1–8 行原值保留，只追加第 9 行。',7,3,9,'CED 全局 KV 来源相同，序列长度向前增长。'],
 ['Decode','第 9 个位置继续经过全部 decoder','第 4 层、5 层、6 层都要运行。每层用当前 Q 读取选中的全局条目，以及本层局部 7、8、9。这里没有“生成阶段也跳过后半模型”。',10,6,9,'本例 decode 的第 5 层在候选池内重新索引。'],
 ['Decode','得到下一次预测所需的末尾状态','第 9 个位置走完整个模型，得到用于预测第 10 个 token 的状态。全局缓存增长到 9 行；各层当前的局部可见位置是 7、8、9。',7,6,9,'逻辑窗口与物理环形缓冲区槽位不是同一个概念。'],
 ['另一请求','1–8 全部命中；从中取出尾部 6–8 重放','这次请求的旧前缀 1–8 与上次完全相同，全局 A、C 中这 8 个位置的缓存全部复用，包括 6–8。这里假设局部 SWA 状态缺失；本例窗口长度为 3，采用有限重放方案，只把旧尾部 6、7、8 再送入模型恢复局部状态。位置 1–5 不重新做逐层前向。',14,0,8,'论文 §3.2.1–3.2.2：全局命中、局部缺失的分支。'],
 ['恢复局部状态','全局 1–8 都复用；只重算尾部的局部状态','上方始终保留完整的旧前缀 1–8。黄色的 6、7、8 从原始嵌入再次经过 encoder，重新准备各层的局部 SWA KV；位置 1–5 不重新前向。右侧绿色全局缓存 1–8 始终保留原值，6–8 对应的全局条目也不覆盖。',10,3,8,'官方 Q&A：replayed prefix 的 global KV 原样复用。'],
 ['恢复局部状态','尾部 6–8 继续经过 decoder，准备局部 KV','重新算出的 6、7、8 的 encoder 表示继续经过 decoder 各层，生成本层的局部 SWA KV。全局 C 的 1–8 仍直接复用。黄色表示这次重新计算的尾部，绿色表示已命中的全局缓存，蓝色表示本次恢复的局部状态。',10,6,8,'论文 §3.2.2：decoder SWA 用于 decode，不写前缀缓存。'],
 ['回到完整流程','复用的是全局 1–8，重放的是旧尾部 6–8','位置 1–8 都属于相同的旧前缀，全局缓存全部复用。位置 1–5 不重新前向；6–8 因为局部 SWA 状态缺失，按本例长度为 3 的窗口重新经过各层。现在局部状态已准备好，可以供后续计算使用。',10,6,8,'可拖动进度回看；详细算式单独保留在下方入口。']
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
 for(let i=0;i<frames.length;i++){const f=frames[i],s=i>=20?restored:i>=14&&i<=18?decoded:short;let v=valFormula(f[4],f[5],s);
 if(i===5)v=String.raw`c_8=h_8^{(3)}P_C=${pair(short.shared.C[8])}`;
 if(i===7||i===8)v=String.raw`q_8^{(4)}=${pair(short.traces[4][8].q)}`;
 if(i===10)v=String.raw`q_8^{(5)}=${pair(short.traces[5][8].q)}`;
 if(i===12)v=String.raw`q_8^{(6)}=${pair(short.traces[6][8].q)}`;
 if(i===16)v=String.raw`c_9=h_9^{(3)}P_C=${pair(decoded.shared.C[9])}`;
 if(i===19)v=String.raw`C_{1:8}\text{ 命中},\quad\mathrm{KV}_{\mathrm{SWA}}\text{ 缺失}`;
 values[i]=await tex(v);
 }
}
function token(i,x,y,label,fill='#f1d6a0',opacity=1){return `<g opacity="${opacity}">${rect(x-25,y-19,50,38,fill,i===8?'#9e6c20':'none')}${text(x,y-1,String(i),17,'','middle')}${text(x,y+13,label,10,'','middle')}</g>`;}
function packet(x1,y1,x2,y2,p,fill='#287864',label=''){const t=ease(Math.min(1,Math.max(0,p))),x=lerp(x1,x2,t),y=lerp(y1,y2,t);return `<g>${rect(x-16,y-12,32,24,fill)}${label?`<text x="${x}" y="${y+4}" style="fill:white" font-size="11" text-anchor="middle">${label}</text>`:''}</g>`;}
function background(i){let s=`<title id="board-title">${esc(frames[i][1])}</title><desc id="board-desc">${esc(frames[i][2])}</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#81998b"/></marker></defs>`;
 s+=rect(16,22,696,262,'#f3f7ef')+rect(16,307,696,204,'#fcf6eb');s+=text(30,43,'ENCODER · 前 3 层',14)+text(30,328,'DECODER · 后 3 层',14);
 for(let l=0;l<=6;l++){s+=text(34,Y[l]+5,l===0?'输入':`第 ${l} 层`,14);if(l>3)s+=text(34,Y[l]+22,['Full','Reindex','Reuse'][l-4],10,'sub');for(let j=1;j<=8;j++){s+=rect(X(j)-26,Y[l]-20,52,40,'none','#dce5d9',5);if(j<6&&i>=19&&l>0)s+=text(X(j),Y[l]+4,'不重算',10,'sub','middle');else if(l>3&&j<6&&i>=6)s+=text(X(j),Y[l]+4,'省去',11,'sub','middle');}}
 s+=text(45,296,'同一列 = 同一个位置；向下 = 经过下一层',13,'sub');
 s+=rect(728,155,255,222,'#eaf4ee','#bbd3c5');s+=text(745,179,i>=19?'全局 C：1–8 全部复用':'全局 C：decoder 共享',16)+text(745,198,i>=19?'旧值不变；6–8 的全局 KV 也复用':'由完整 encoder 输出投影',12,'sub');
 if(i>=19){s+=rect(728,22,255,114,'#f3f7ef','#bbd3c5');s+=text(742,45,'相同旧前缀：位置 1–8',15)+text(742,70,'全局 A / C：1–8 均命中',13)+text(742,94,'局部 SWA：缺失 → 重放 6–8',12)+text(742,119,'本例窗口为 3；尾部仍经过各层',12,'sub');}
 s+=rect(728,385,255,183,'#f0f5f7','#c5d3d7');s+=text(745,403,'各层自己的局部 KV',14);s+=text(741,590,'Encoder 全局 A 未展开；与 C 分开。',11,'sub');
 return s;
}
function localLevel(i,p){if(i<2)return 0;if(i<=4)return i-1;if(i<=6)return 3;if(i<=9)return 4;if(i<=11)return 5;if(i<19)return 6;if(i===19)return 0;if(i===20)return Math.min(3,Math.floor(p*3)+1);return i===21?3+Math.min(3,Math.floor(p*3)+1):6;}
function draw(i,p){let s=background(i),progress=reduced?1:ease(Math.min(1,p/.55)),localDone=localLevel(i,p),numC=i<5?0:(i>=16&&i<=18?9:8);
 for(let j=1;j<=9;j++){const [x,y]=C(j);let visible=j<=numC&&(i!==5||p>.55)&&!(j===9&&i===16&&p<.55);let selected=[];
 if([8,9].includes(i))selected=short.traces[4][8].selected;if(i>=10&&i<=13)selected=short.traces[i<12?5:6][8].selected;if(i===17)selected=decoded.traces[Math.min(6,4+Math.floor(p*3))][9].selected;
 s+=rect(x-25,y-15,50,32,visible?(selected.includes(j)?'#9ccdb3':'#d8eadd'):'#f8fbf8',selected.includes(j)?'#287864':'#cadbd0',4)+text(x,i>=19&&visible?y-1:y+5,visible?`C · ${j}`:j===9?'待追加':'空',12,visible?'':'sub','middle');if(i>=19&&visible)s+=text(x,y+12,'已复用',9,'','middle');}
 for(let l=1;l<=6;l++){s+=text(742,loc(l,6)[1]+4,`层 ${l}`,11,'sub');for(let j=6;j<=8;j++){const [x,y]=loc(l,j),show=l<=localDone;const written=i===15?l<=Math.min(3,Math.floor(p*3)+1):i===16?l<=3:i===17?l<=Math.min(6,4+Math.floor(p*3)):i===18;const slot=written?j+1:j;s+=rect(x-16,y-9,32,20,show?'#c8dce1':'#fff',show?'none':'#d4dfe1',3)+text(x,y+5,show?String(slot):'—',11,'','middle');}}
 // Fixed historical outputs stay on the board; moving tokens are the active computation.
 if(i>=5&&i<=18){for(let j=1;j<=8;j++)s+=token(j,X(j),Y[3],'编码后','#e5d4b1',.25);}
 let range=positions(1,8),y=Y[0],label='token',fill='#f1d6a0';
 if(i===1)label='嵌入';
 if(i>=2&&i<=4){const l=i-1;y=lerp(Y[l-1],Y[l],progress);label=progress<.8?'计算中':`层${l}输出`;}
 if(i===5){y=Y[3];label='层3输出';for(let j=1;j<=8;j++){const [cx,cy]=C(j);s+=path(`M${X(j)} ${Y[3]} Q710 120 ${cx} ${cy}`,'global-wire',false);s+=packet(X(j),Y[3],cx,cy,progress,'#287864',String(j));}}
 if(i===6){y=Y[3];label='继续';for(let j=1;j<=5;j++)s+=token(j,X(j),y,'到此停止','#e5e8df',1-progress);range=[6,7,8];}
 if(i>=7&&i<=13){range=[6,7,8];let l=i<=9?4:i<=11?5:6;const output=[9,11,13].includes(i);y=output?lerp(Y[l-1],Y[l],progress):Y[l-1];label=output?`层${l}输出`:'本层输入';
 if([7,8,10,12].includes(i)){s+=rect(98,519,608,83,'#f8ecd5','#ddc6a0');s+=text(116,541,`第 ${l} 层注意力：主 Q + 局部 KV + 全局 KV → 更新状态`,14);s+=token(8,160,574,'本层 Q');
 const selected=short.traces[l][8].selected;
 s+=path(`M${X(8)} ${Y[l-1]+20} L${X(8)} 515 L160 515 L160 552`);
 if(i!==7){for(const j of selected){const [cx,cy]=C(j),endX=485+selected.indexOf(j)*60;s+=path(`M${cx} ${cy} Q718 510 ${endX} 574`,'global-wire',false);s+=packet(cx,cy,endX,574,progress,'#287864',`C${j}`);}for(let j=6;j<=8;j++){const [lx,ly]=loc(l,j);s+=packet(lx,ly,270+(j-6)*52,574,progress,'#789ca6',String(j));}}
 else{for(let j=6;j<=8;j++){const [lx,ly]=loc(l,j);s+=packet(X(j),Y[l-1],lx,ly,progress,'#789ca6',String(j));}}
 }
 }
 if(i>=14&&i<=18){range=[];for(let j=6;j<=8;j++)s+=token(j,X(j),Y[6],'旧状态','#e6d7b9',.45);
 let newY=Y[0],newLabel='新 token';
 if(i===14){s+=rect(125,524,550,64,'#f8ecd5');s+=text(145,547,'末尾状态 → 输出头 / 采样（本例省略）',16);s+=text(145,574,'外部指定新 token：“蓝鲸”，位置编号 9',16);s+=path(`M${X(8)} ${Y[6]} Q720 520 ${X(9)} ${Y[0]}`);newY=lerp(Y[6],Y[0],progress);}
 if(i===15){const t=Math.min(2.999,p*3),l=Math.floor(t)+1;newY=lerp(Y[l-1],Y[l],reduced?1:ease(t%1));newLabel=`层${l}计算`;}
 if(i===16){newY=Y[3];newLabel='层3输出';const [cx,cy]=C(9);s+=path(`M${X(9)} ${Y[3]} L${cx} ${cy}`,'global-wire');s+=packet(X(9),Y[3],cx,cy,progress,'#287864','C9');}
 if(i===17){const t=Math.min(2.999,p*3),l=4+Math.floor(t);newY=lerp(Y[l-1],Y[l],reduced?1:ease(t%1));newLabel=`层${l}计算`;const selected=decoded.traces[l][9].selected;for(const j of selected){const [cx,cy]=C(j);s+=packet(cx,cy,X(9),newY,ease(t%1),'#287864',`C${j}`);}s+=text(120,550,`第 ${l} 层：局部 {7,8,9}；全局 {${selected.join(',')}}`,17);}
 if(i===18){newY=Y[6];newLabel='层6输出';s+=text(120,550,'这个末尾状态将用于下一次预测第 10 个 token。',18);}
 s+=token(9,X(9),newY,newLabel,'#f1d6a0');s+=text(X(9),35,'新位置',12,'','middle');
 }
 if(i>=19){
 // Keep the entire matching prefix visible; moving packets show only recomputation.
 for(let j=1;j<=8;j++)s+=token(j,X(j),Y[0],TOKENS[j-1],'#d8eadd');
 s+=rect(X(6)-28,Y[0]-22,X(8)-X(6)+56,44,'none','#b08030',6);
 s+=text(X(3),99,'1–5：不重新前向',11,'sub','middle')+text(X(7),99,'6–8：取出重放',11,'','middle');
 range=i===19?[]:[6,7,8];let l=0;y=Y[0];label='旧尾部';
 if(i===20||i===21){const t=Math.min(2.999,p*3);l=(i===20?1:4)+Math.floor(t);y=lerp(Y[l-1],Y[l],reduced?1:ease(t%1));label=`重算层${l}`;}
 if(i===22){y=Y[6];label='已恢复';}
 s+=text(118,544,'复用范围：全局 1–8（包括 6–8）',17)+text(118,575,i===19?'重放范围：只取旧尾部 6–8，准备缺失的局部 KV':'绿色旧全局缓存不覆盖；黄色尾部重新计算，恢复蓝色局部 KV。',15);
 }
 for(const j of range)s+=token(j,X(j),y,i===0?TOKENS[j-1]:label,fill);
 if(i===0||i===1)s+=text(118,553,'先认编号：后面追踪的始终是同一个位置，不是换了一批 token。',17);
 $('board').innerHTML=s;
}
function showSelection(i){
 const panel=$('selection-flow'),l=i===8?4:i===10?5:i===12?6:0;
 panel.hidden=!l;if(!l){panel.innerHTML='';return;}
 const trace=short.traces[l][8],reuse=l===6,chosen=trace.selected.join('、');
 const scores=trace.scores.map(row=>`<li class="${trace.selected.includes(row.pos)?'chosen':''}"><span>位置 ${row.pos}</span><strong>${row.score.toFixed(3)}</strong><small>${trace.selected.includes(row.pos)?'选中':'未选'}</small></li>`).join('');
 panel.innerHTML=`<h3>位置 8 · 第 ${l} 层：${reuse?'复用索引，重新计算注意力':'先选地址，再读取内容'}</h3>
 <ol class="selection-steps">
 <li><strong>① ${reuse?'跳过重新评分':'索引器评分'}</strong><p>${reuse?'Reuse 层沿用上一层的结果，不生成新的索引查询，也不重算这些分数。':`本层输入 → 索引查询 → 对可见位置 ${trace.scanned[0]}–${trace.scanned.at(-1)} 评分。索引查询与主注意力 Q 是不同的投影。`}</p></li>
 <li><strong>② ${reuse?'沿用位置索引':'Top-2 选位置索引'}</strong><p>${reuse?'直接接收上一层的索引':'保留评分最高的两个位置'}：<b>${chosen}</b>。这里得到的是位置编号，尚未得到注意力输出。</p></li>
 <li><strong>③ 主注意力读取 KV</strong><p>按索引读 C 中的 <b>${chosen}</b>，再加上本层局部 <b>${trace.localPositions.join('、')}</b>。用本层主 Q 计算注意力分数、归一化并加权汇总。</p></li></ol>
 ${reuse?'':`<p class="score-label">本例索引评分（不是最终注意力权重；按位置排列，绿色标出 Top-2）</p><ul class="index-scores">${scores}</ul>`}
 <p class="selection-note">局部位置由滑动窗口确定。本例用 C 简化索引键；真实模型有独立的 indexer K。Top-2 与这些分数只用于教学。<a href="https://arxiv.org/html/2609.19969v1#S2.S3.S1" target="_blank" rel="noopener">论文 §2.3.1 ↗</a></p>`;
}
function status(i){if(i<5)return '全局 C 尚未构造。encoder 的局部状态逐层建立。';if(i<14)return 'C：位置 1–8 保留。局部 KV：按层独立保存。';if(i<16)return '读取旧缓存，开始为新位置生成状态。';if(i<19)return 'C：追加至位置 9。局部窗口：7、8、9。';return '全局 1–8 全部复用；1–5 不重新前向，6–8 重放恢复局部 KV。';}
function update(t,force=false){time=Math.min(duration,Math.max(0,t));const [i,p]=at(time);
 if(i!==frame||force){frame=i;showSelection(i);const f=frames[i];$('phase').textContent=f[0];$('frame-title').textContent=f[1];$('narration').textContent=f[2];$('math-value').innerHTML=values[i];$('tracking').textContent=`位置 ${f[5]} · ${TOKENS[f[5]-1]}`;$('cache-status').textContent=status(i);$('source-label').textContent=f[6];$('frame-count').textContent=`第 ${i+1} / ${frames.length} 页`;$('prev').disabled=i===0;$('next').disabled=i===frames.length-1;
 const group=i<5?0:i<14?5:i<19?14:19;document.querySelectorAll('[data-jump]').forEach(b=>b.setAttribute('aria-current',String(Number(b.dataset.jump)===group)));
 }
 $('timeline').value=Math.min(frames.length,i+p);$('timeline').setAttribute('aria-valuetext',`第 ${i+1} 幕：${frames[i][1]}`);draw(i,p);
}
function stop(){playing=false;cancelAnimationFrame(raf);$('play').textContent=time>=duration?'重新放映':'自动放映';$('play').setAttribute('aria-pressed','false');}
function tick(now){if(!playing)return;if(last)update(time+Math.min((now-last)/1000,.2)*speed);last=now;if(time>=duration){stop();return;}raf=requestAnimationFrame(tick);}
function play(){if(!ready)return;cancelAnimationFrame(raf);if(time>=duration)update(0);playing=true;last=0;$('play').textContent='暂停放映';$('play').setAttribute('aria-pressed','true');raf=requestAnimationFrame(tick);}
function jump(i){stop();const target=Math.max(0,Math.min(frames.length-1,i));update(starts[target],true);if(reduced){update(starts[target]+frames[target][3]*.99);return;}let began;function pageTurn(now){began??=now;const p=Math.min(1,(now-began)/2200);update(starts[target]+frames[target][3]*.99*p);if(p<1)raf=requestAnimationFrame(pageTurn);}raf=requestAnimationFrame(pageTurn);}
async function init(){await MathJax.startup.promise;await prepare();ready=true;$('timeline').max=String(frames.length);
 const hash=location.hash.slice(1),aliases={input:0,encoder:2,fork:5,full:7,reindex:10,reuse:12,replay:19,decode:14,pool:17,restore:19,summary:22};
 update(starts[aliases[hash]??0],true);$('play').setAttribute('aria-pressed','false');
 $('play').addEventListener('click',()=>playing?stop():play());$('restart').addEventListener('click',()=>{stop();update(0,true);play();});$('prev').addEventListener('click',()=>jump(frame-1));$('next').addEventListener('click',()=>jump(frame+1));
 $('speed').addEventListener('change',()=>{speed=Number($('speed').value);});$('timeline').addEventListener('input',()=>{stop();const v=Number($('timeline').value),i=Math.min(frames.length-1,Math.floor(v));update(v>=frames.length?duration:starts[i]+(v-i)*frames[i][3],true);});
 document.querySelectorAll('[data-jump]').forEach(b=>b.addEventListener('click',()=>{jump(Number(b.dataset.jump));}));
 $('zoom').addEventListener('click',()=>{const on=$('screen').classList.toggle('zoomed');$('zoom').setAttribute('aria-pressed',String(on));$('zoom').textContent=on?'适应宽度':'放大画面';});
 document.addEventListener('keydown',e=>{if(e.target.closest('input,select,textarea,a'))return;if(e.key==='ArrowRight'){e.preventDefault();jump(frame+1);}if(e.key==='ArrowLeft'){e.preventDefault();jump(frame-1);}});
 document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
 if(reduced)$('motion-note').textContent='已尊重“减少动态效果”设置，可逐幕查看。';
 $('board').dataset.ready='true';
}
init().catch(e=>{$('error').hidden=false;console.error(e);});
