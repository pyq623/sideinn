import {draftDiagram,fitDraftDiagram} from './draft-diagram.mjs?v=20261023-2';
import {requests,rank,survival,budgets,best,allocate,verification,acceptance,residual,markov,markovQ,bonus,corrected,added,correctedLaw} from './model.mjs';
const $=id=>document.getElementById(id), f=(n,d=3)=>Number(n.toFixed(d)).toString(), tex=s=>`<div class="math">\\(${s}\\)</div>`;
const paper='https://arxiv.org/html/2607.05147v1', code='https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/';
const sources=[['DSpark §3.1 · 并行骨干与顺序头',paper+'#S3.S1'],['DeepInfer · 历史预算与长度分配',code+'csrc/engine/mtp_varlen.hpp'],['投机采样原论文 · Algorithm 1','https://arxiv.org/html/2211.17192#S2.S3'],['DeepInfer · 采样与首个拒绝位置',code+'deepinfer/model_executor/layers/tilelang_ops/sampler.py']];
const panel=(title,body)=>`<div class="panel enter"><h3>${title}</h3>${body}</div>`;
const flow=items=>`<div class="flow">${items.map((item,i)=>`<div class="enter" style="--delay:${i*.2}s"><span class="tag">${i+1 < 10?'0':''}${i+1}</span>${item}</div>`).join('')}</div>`;
const table=(heads,rows)=>`<div class="table-wrap"><table><thead><tr>${heads.map(x=>`<th>${x}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
const row=(cells,cls='')=>`<tr class="${cls}">${cells.map(x=>`<td>${x}</td>`).join('')}</tr>`;
const token=(label,sub='',cls='')=>`<span class="token ${cls}">${label}<small>${sub}</small></span>`;
const ranked=rank();
const rankView=n=>panel('按前缀存活率排队：选中的名额逐个亮起',`<div class="rank-list">${ranked.map((r,i)=>`<div class="rank-item enter ${i<n?'active':''}" style="--delay:${i*.1}s"><span>${r.id}${r.j}</span><strong>${f(r.a)}</strong><small>${i<n?'已分配':'待分配'}</small></div>`).join('')}</div><p class="note">同分时按请求顺序、位置顺序稳定排序。后一个位置不能越过同一请求的前一个位置。</p>`);
function budgetSymbolGuide(){
 const totalRequests=requests.length,allocation=allocate(best.total),lengths=allocation.lengths;
 return `<div class="budget-symbol-guide"><h3>先把公式中的符号，对回这两条请求</h3><div class="budget-symbol-grid"><section><div class="budget-symbol">${tex(`R=${totalRequests}`)}</div><strong>R 是请求数量</strong><p>请求 A、请求 B，一共 <b>${totalRequests} 条请求</b>。这里数的是请求，不是 token。</p><div class="request-chips"><span>请求 A</span><span>请求 B</span></div></section><section><div class="budget-symbol">${tex(`B=${best.total}`)}</div><strong>B 是本轮目标前向的 query token 总数</strong><p>公式里的 <b>B 表示计算规模</b>，不是那条名字叫“B”的请求。每条请求输入 1 枚 anchor，再加选中的草稿。</p><div class="budget-inputs">${requests.map((r,k)=>`<div><span>请求 ${r.id}：</span><b>${r.anchor}<small>anchor</small></b>${r.draft.slice(0,lengths[k]).map(w=>`<b>${w}<small>草稿</small></b>`).join('')}</div>`).join('')}</div>${tex(`B=(1+${lengths[0]})+(1+${lengths[1]})=${best.total}`)}</section></div><div class="budget-symbol-note">小写 r 指哪条请求，j 指它的第几枚草稿；${tex(String.raw`a_{r,j}`)} 是上图该草稿的前缀存活率。${tex(String.raw`t(B)`)} 是处理 B 个 query 的本轮耗时；${tex(String.raw`\Theta(B)`)} 是预计每秒新增多少 token。</div><div class="budget-r-reason"><strong>为什么分子里也加 R？</strong><p>每条请求除了接受的草稿，还会产生 <b>1 枚纠正或追加 token</b>，因此 ${totalRequests} 条请求先贡献 ${totalRequests} 枚。再加上选中草稿的存活率之和，得到期望新增数。这里的 R <b>不是把旧 anchor 再算成新输出</b>；本例忽略 EOS 和长度上限。</p></div></div>`;
}
function budgetSubstitution(){
 const allocation=allocate(best.total),gains=allocation.selected.map(x=>f(x.a));
 return `<div class="budget-substitution"><h3>把绿色行的数字代进去</h3><p>选中 A1、A2、B1，分别贡献 ${gains.join('、')} 的预期收益：</p>${tex(String.raw`\begin{aligned}R+\sum a_{r,j}&=${requests.length}+${gains.join('+')}\\&=${f(allocation.expected)}\end{aligned}`)}<p>这是假设每轮新增 ${f(allocation.expected)} 枚、每轮耗时 ${best.ms} ms；乘 1000 把“每毫秒”换成“每秒”：</p>${tex(`\\Theta(${best.total})=\\frac{${f(allocation.expected)}}{${best.ms}}\\times1000=${Math.round(best.tps)}`)}<p class="budget-answer">预计吞吐：<strong>${Math.round(best.tps).toLocaleString('en-US')} token/s</strong>。B = ${best.total} 是输入计算量，${f(allocation.expected)} 是期望新增输出量，两者含义不同。</p></div>`;
}
const costView=(all=false)=>panel('用历史估计查询假设成本表',table(['总 query 数 B','请求 A / 请求 B 草稿数','期望新增 token','耗时 t(B)','预计吞吐 Θ(B)'],budgets.filter((_,i)=>all||i===0||i===3||i===6).map(r=>row([r.total,r.lengths.join(' / '),f(r.expected),`${r.ms} ms`,Math.round(r.tps).toLocaleString('en-US')],r.total===best.total?'chosen':'')))+`<p class="note">成本表为教学设定，包含本轮开销；没有执行 GPU benchmark。新增长度含纠正／追加的 1 枚 token，不含上一轮已经交付的 anchor。</p>`);
function meter(a,u){return `<div class="decision-meter" aria-label="接受概率 ${f(a)}，随机数 ${u}"><div class="meter-track"><i style="width:${a*100}%"></i><b style="left:${u*100}%"><span>u = ${u}</span></b></div><div class="meter-labels"><span>0</span><span>绿色区间：u 小于 α 时接受</span><span>1</span></div></div>`;}
function distribution(key,decision='') {const v=verification[key],a=acceptance(v.p,v.q,v.candidate);return panel(`${key} · 在相同前缀下比较两种分布`,table(['非零词表项','草稿 q','目标 p','本轮候选'],v.vocab.map((w,i)=>row([w,f(v.q[i]),f(v.p[i]),i===v.candidate?'← '+w:''],i===v.candidate?(decision&&v.u>a?'rejected':'candidate'):'')))+tex(`\\alpha=\\min(1,\\frac{${f(v.p[v.candidate])}}{${f(v.q[v.candidate])}})${a===1?'=1':'\\approx ' + f(a)}`)+meter(a,v.u)+`<div class="emphasis">${decision||'先得到完整概率表，再计算这枚候选的接受概率。'}</div><p class="note">共同条件：${v.context}。u = ${v.u} 是本轮演示固定的均匀随机数。</p>`);}
const frames=[
 {title:'先固定这两条请求，接下来一直跟着它们走',text:'A 正在回答“请执行操作”，B 正在打招呼。上一轮已经给出“先”和“说”，它们是本轮的 anchor（锚点）。本轮需要继续写，先让便宜的草稿模型各提议 3 个 token。',view:()=>flow(['<strong>已经确定的前缀</strong>旧内容有缓存；最后一枚 anchor 将作为这一轮的输入。','<strong>先写草稿，再分名额</strong>草稿只是候选，还不能直接发给用户。','<strong>目标模型把关</strong>本轮交付的文字必须经过采样接受或纠正。']),take:'旧前缀 → 草稿 → 调度 → 验证 → 新的确定前缀。',math:String.raw`R=2,\qquad \gamma=3`,source:0},
 {title:'昂贵的草稿骨干：三个位置一起准备',text:'以 A 为例，DSpark 的并行骨干接收目标模型的上下文特征，以及 anchor 加两个 mask，一次生成 3 组基础表示和 logits。此时还没有确定三个位置最终选哪个词。',view:()=>draftDiagram('parallel'),take:'并行的是大部分草稿计算；采样后的词间依赖留给轻量顺序头。',math:String.raw`(h_1,U_1),\ (h_2,U_2),\ (h_3,U_3)`,source:0},
 {title:'轻量顺序头：先选出 A1 =「打开」',text:'顺序头从左向右采样。这里第一位置的草稿分布是“打开 0.6、运行 0.3、跳过 0.1”。用演示随机数 0.2 落入第一段，得到“打开”；它随后影响第二位置。',view:()=>draftDiagram('first')+panel('从 q₁ 采样，不是挑最大概率的词',`<div class="prob-grid">${verification.A1.vocab.map((w,i)=>`<div class="prob-cell enter" style="--delay:${i*.2}s">${w}<strong>${f(verification.A1.q[i])}</strong></div>`).join('')}</div><p class="emphasis">区间 [0, 0.6) → 打开；[0.6, 0.9) → 运行；[0.9, 1) → 跳过。随机数 0.2 选择“打开”。</p>`),take:'A1 已经选定，A2 的轻量修正会用到“打开”。这仍只是草稿。',math:String.raw`x_{A,1}\sim q_{A,1},\qquad x_{A,1}=\text{打开}`,source:0},
 {title:'A1 改变 A2 的分布，再顺序写完草稿',text:'A2 的基础分布是“文件 0.4、测试 0.3、目录 0.3”。本例给“打开”设定一个转移偏置，加到基础 logits 后再 softmax，得到 0.2、0.6、0.2；用随机数 0.55 选出“测试”。随后写出 A3。B 也走同样过程。',view:()=>draftDiagram('complete')+panel('只演算 A2 的轻量修正',table(['词','基础概率','加到 logits 的偏置','修正后的 q₂'],markovQ.map((q,i)=>row([verification.A2.vocab[i],markov.base[i],f(markov.afterOpen[i]),f(q)])))+`<p class="note">偏置为人工设定的向量，用来展示 Markov 头的作用；真实实现通过低秩矩阵生成偏置。这里没有复现矩阵参数。</p>`),take:'“打开 → 测试 → 。”是本次抽到的草稿；草稿连贯性改善不等于保证被接受。',math:String.raw`q_2=\operatorname{softmax}(\log b_2+\Delta_{\text{打开}})=(0.2,0.6,0.2)`,source:0},
 {title:'再准备调度用的估计：c 不是 q，也不是验收结果',text:'q 描述“会抽到哪个词”；c 估计“这个位置在前面都通过时，通过验证的概率”。调度时还不知道本轮目标概率 p。本例使用历史置信度快照：A 为 0.9、0.9、0.9，B 为 0.8、0.5、0.5；它们不是稍后实际验收的答案。',view:()=>flow(['<strong>q：草稿分布</strong>每个位置是一整张词表分布，用来生成候选。','<strong>c：置信度估计</strong>每个位置一个数，用来估计验证收益。','<strong>p：目标分布</strong>下一阶段目标模型运行后才得到，用于正式验收。']),take:'下面的调度只看可用的历史估计，不偷看本轮目标验证结果。',math:String.raw`c_A=(0.9,0.9,0.9),\quad c_B=(0.8,0.5,0.5)`,source:1},
 {title:'先累乘：A3 有价值，必须先让 A1、A2 存活',text:'A3 的条件通过率估计是 0.9，但最终保留下来的概率估计只有 0.9 × 0.9 × 0.9 = 0.729。对 B 同样累乘，得到 0.8、0.4、0.2。这个“前缀存活率”才是多给一个验证名额的预期收益。',view:()=>panel('同一组估计，先算成前缀收益',table(['请求','第 1 枚','第 2 枚','第 3 枚'],requests.map(r=>row([r.id,...survival(r.confidence).map((a,j)=>tex(`${r.confidence.slice(0,j+1).join('\\times')} = ${f(a)}`))])))),take:'后面的 token 即使单独容易通过，也要乘上前面全部通过的机会。',math:String.raw`a_{r,j}=\prod_{i=1}^{j}c_{r,i}`,source:1},
 {title:'决定总预算：比“每秒能推进多少”，不只比接受长度',text:'R 是请求数量：A、B 两条请求，所以 R = 2。B 是本轮目标前向处理的 query token 总数：每条请求的 anchor，加上被选中的草稿。下表逐行尝试不同的 B；注意公式里的 B 与“请求 B”的名字无关。用较早轮次的存活率和假设成本表，比较每秒预计新增多少 token。',view:()=>costView(true),take:'绿色行：总共算 5 个 query，预计推进 4.51 个 token，假设耗时 2 ms。',math:String.raw`\Theta(B)=\frac{R+\sum_{\text{选中草稿}}a_{r,j}}{t(B)}\times1000`,guide:budgetSymbolGuide,example:budgetSubstitution,source:1},
 {title:'预算确定为 5：其中只有 3 个是草稿名额',text:'本例成本表中，5 个 query 的预计吞吐最高，为 2255 token/s。全验 6 枚草稿需要 8 个 query，虽然预计推进 5.839 个 token，但耗时 3 ms，只有约 1946 token/s。现在锁定预算，再用较新轮次的估计分给 A、B。',view:()=>flow(['<strong>较早轮次估计</strong>配合成本表，选择总预算 5。','<strong>扣掉两个基础 query</strong>5 − 2 = 3 个额外名额。','<strong>较新轮次估计</strong>把 3 个名额分给两个请求的前缀。']),take:'定总预算和分配每条请求的长度，是两个决定；本例没有用本轮未来词倒推预算。',math:String.raw`B=R+\ell_A+\ell_B=2+2+1=5`,source:1},
 {title:'第一个名额给 A1：收益估计 0.9',text:'把所有位置按前缀存活率从高到低排列。A1 的 0.9 最大，先拿走一个名额；这表示值得花一次验证计算，还没有宣布“打开”通过。',view:()=>rankView(1),take:'当前分配：A 验 1 枚，B 验 0 枚；还剩 2 个名额。',math:String.raw`(\ell_A,\ell_B)=(1,0)`,source:1},
 {title:'第二个名额给 A2：0.81 高于 B1 的 0.8',text:'A2 的收益是 0.9 × 0.9 = 0.81，略高于 B1。A1 已经入选，因此选 A2 仍是连续前缀，不会出现跳过 A1、单独验证 A2 的情况。',view:()=>rankView(2),take:'当前分配：A 验 2 枚，B 验 0 枚；还剩 1 个名额。',math:String.raw`(\ell_A,\ell_B)=(2,0)`,source:1},
 {title:'第三个名额给 B1，剩余草稿在调度阶段停下',text:'B1 的 0.8 高于 A3 的 0.729，所以最后一个名额归 B1。最终 A 留“打开、测试”，B 留“你好”。A3、B2、B3 没有送去验证；灰色表示“未分到名额”，不表示目标模型拒绝了它们。',view:()=>rankView(3),take:'固定结果：A 验 2 枚、B 验 1 枚。接下来才真正运行目标模型。',math:String.raw`\mathbb{E}[\Delta L_A+\Delta L_B]=2+0.9+0.81+0.8=4.51`,source:1},
 {title:'把选中的前缀打包成一次目标前向',text:'A 输入“先、打开、测试”共 3 个 query；B 输入“说、你好”共 2 个 query。旧前缀通过各自 KV 缓存参与计算。两条请求可以放在同一个批次，但注意力彼此隔离；基础 query 是已确定的 anchor，不是新输出。',view:()=>panel('这就是总 query 数 5 的实际含义',`<div class="grid"><div><h3>A · 3 个 query</h3><div class="tokens">${token('先','anchor','anchor')}${token('打开','A1')}${token('测试','A2')}</div></div><div><h3>B · 2 个 query</h3><div class="tokens">${token('说','anchor','anchor')}${token('你好','B1')}</div></div></div><div class="signal"></div><div class="emphasis">旧前缀 KV → 目标模型的一次批量前向 → A 的 3 组分布 + B 的 2 组分布</div>`),take:'处理 anchor 的那一行预测第一枚草稿；处理最后草稿的那一行预测可能追加的 token。',math:String.raw`B=(1+2)+(1+1)=5`,source:2},
 {title:'一行错开一位：看到“先”，才预测“打开”',text:'目标模型是因果模型。A 的第一行只看到旧前缀和“先”，输出用来检验 A1 的 p₁；第二行再看到“打开”，输出检验 A2 的 p₂。多个位置能一起前向，但前面的分布不能看后面的候选。',view:()=>panel('A 的目标前向：输入行与输出分布对齐',table(['本行输入','允许看到的新增前缀','本行输出用途'],[row(['先','先','p₁：检验「打开」']),row(['打开','先、打开','p₂：检验「测试」']),row(['测试','先、打开、测试','p₃：仅在 A1、A2 全接受后用来追加'])])+`<p class="emphasis">因果遮罩挡住右侧未来位置。并行得到概率，不等于可以越过首个拒绝位置交付。</p>`),take:'p₂ 的条件里有“打开”，但没有正在被检验的“测试”。',math:String.raw`p_2(v)=P_{\mathrm{target}}(v\mid\text{旧前缀},\text{先},\text{打开})`,source:2},
 {title:'现在才有 p：先看 A1 的完整概率表',text:'目标模型对 A1 给出“打开 0.8、运行 0.1、跳过 0.1”。草稿当时从 q₁ 抽到了“打开”。验证比较的是同一上下文、同一候选的 p/q，不是再让目标模型独立抽一个词，看文字是否一致。',view:()=>distribution('A1'),take:'本页的 p 和 q 是人工分布，用来执行真实的采样接受规则。',math:String.raw`\alpha(x)=\min\!\left(1,\frac{p(x)}{q(x)}\right),\quad u\sim U[0,1)`,source:2},
 {title:'A1 通过：“打开”进入接受前缀',text:'“打开”的 p/q = 0.8/0.6，大于 1，所以接受概率截为 1。固定随机数 0.4 小于 1，本轮接受 A1。随后才允许把接受前缀向 A2 延伸。',view:()=>distribution('A1','u = 0.4 < α = 1 → 接受「打开」。'),take:'A 当前已接受 1 枚草稿；这一步不再额外抽一个目标 token。',math:String.raw`\alpha_{A,1}=1,\qquad u_{A,1}=0.4<1`,source:2},
 {title:'A2 拒绝：有验证名额，也不保证通过',text:'“测试”的草稿概率是 0.6，目标概率只有 0.2。接受概率是 1/3，而本次随机数 0.8 更大，因此拒绝 A2。之前调度用的置信度 0.9 是估计；不能拿它代替这里的 p/q。',view:()=>distribution('A2','u = 0.8 > α = 1/3 → 拒绝「测试」，停止延伸接受前缀。'),take:'现在保留“打开”，但“测试”不能交付；还要在这个位置采样纠正 token。',math:String.raw`\alpha_{A,2}=\frac{0.2}{0.6}=\frac13<0.8`,source:2},
 {title:'怎样纠正？把 p − q 的正部分归一化',text:'同一个位置上，“文件”的 p 比 q 多 0.5；其余两项都是负差，截成 0。正残差归一化后，纠正分布全部落在“文件”。于是 A 本轮确定的新内容为“打开、文件”。',view:()=>panel('A2 的纠正分布，逐项计算',table(['词','p − q','取正部分','归一化'],verification.A2.vocab.map((w,i)=>row([w,f(verification.A2.p[i]-verification.A2.q[i]),f(Math.max(0,verification.A2.p[i]-verification.A2.q[i])),f(residual(verification.A2.p,verification.A2.q)[i])],i===0?'chosen':'')))+`<div class="tokens">${token('打开','草稿接受','accepted')}<span class="arrow">→</span>${token(corrected,'残差纠正','sampled')}</div>`),take:'纠正通常需要从残差分布随机采样；本例恰好只有“文件”的残差非零。',math:String.raw`p_{\mathrm{res}}(v)=\frac{[p(v)-q(v)]_+}{\sum_w[p(w)-q(w)]_+}=(1,0,0)`,source:2},
 {title:'拒绝以后，旧分支上的结果不能接着用',text:'A 的 p₃ 是在“先、打开、测试”这个错误分支上算的；现在前缀变成“先、打开、文件”，所以不能用那个 p₃ 追加 token。A3“。”早在调度时就没入选，它与这次验证拒绝是两件事。',view:()=>flow(['<strong>保留</strong>“先、打开”的有效前缀及对应状态。','<strong>回退／失效</strong>“测试”分支的后续采样结果、无效缓存状态不能继续使用。','<strong>下一轮起点</strong>把纠正出来的“文件”作为新 anchor，继续向前。']),take:'算出来不代表能用：只有条件前缀仍然成立的结果才可延续。',math:String.raw`p(\cdot\mid\text{先,打开,测试})\ne p(\cdot\mid\text{先,打开,文件})\ \text{（一般情形）}`,source:3},
 {title:'看另一条请求：B1 =「你好」也通过',text:'B 只分到一枚草稿。“你好”的 q = 0.5、p = 0.6，接受概率为 1，本轮通过。注意“全部接受”只指本轮安排验证的前缀全部接受，并不包含已经裁掉的“朋友、！”两个草稿。',view:()=>distribution('B1','u = 0.4 < α = 1 → 接受「你好」。B 的已调度草稿已全部通过。'),take:'B 接受了一枚草稿，接着还可以从目标分布再采样一枚。',math:String.raw`\ell_B=1,\quad n_B=1=\ell_B`,source:2},
 {title:'B 全部通过，再从已算出的 p₂ 追加一枚',text:'B 的第二行输入是“你好”，已在同一次前向中得到下一位置的目标分布。这里“朋友 0.1、句号 0.8、叹号 0.1”，随机数 0.5 采到“。”。这枚追加 token 不是把被裁掉的草稿 B2 重新拿回来。',view:()=>panel('不需要为了 bonus 再跑一次目标前向',`<div class="prob-grid">${bonus.vocab.map((w,i)=>`<div class="prob-cell enter" style="--delay:${i*.2}s">${w}<strong>${f(bonus.p[i])}</strong></div>`).join('')}</div><div class="tokens">${token('你好','草稿接受','accepted')}<span class="arrow">→</span>${token(added,'目标追加','sampled')}</div><p class="note">如果第一次就拒绝，也会在首位置纠正，仍至少产生一枚；本例忽略 EOS 和长度限制。</p>`),take:'两条分支都多出 1 枚：遇到拒绝则纠正，全接受则从末尾目标分布追加。',math:String.raw`x_{B,2}\sim p_2(\cdot\mid\text{旧前缀},\text{说},\text{你好})`,source:2},
 {title:'交付结果：这一轮共新增 4 枚 token',text:'A 交付“打开、文件”，B 交付“你好、。”。两个旧 anchor“先、说”已经在上一轮交付，不能再计入本轮新增数。我们花了 5 个 query，但这一次实际新增 4 枚，不是 5 枚。',view:()=>panel('把预计收益与这次结果分开',table(['请求','接受草稿','纠正／追加','实际新增'],[row(['A','打开','文件（纠正）',2]),row(['B','你好','。（追加）',2]),row(['合计','2 枚','2 枚',4],'chosen')])+`<p class="emphasis">预计 4.51 是许多轮次意义上的估计值；这一次随机结果是 4。预测可以有误差，单轮也有波动。</p>`),take:'一个批次的目标前向，完成两条请求各 2 枚新 token 的交付。',math:String.raw`\Delta L_A+\Delta L_B=(1+1)+(1+1)=4`,source:2},
 {title:'下一轮接着哪里开始？从新的确定前缀开始',text:'A 的新 anchor 是“文件”，B 的新 anchor 是“。”。旧草稿后缀不再直接沿用；下一轮围绕已确认前缀重新提议候选，并按引擎时序更新置信度和耗时估计。箭头回到第一节，才形成完整解码循环。',view:()=>flow(['<strong>A：先 → 打开 → 文件</strong>接受前缀保留；“文件”成为下一轮 anchor。','<strong>B：说 → 你好 → 。</strong>“。”成为下一轮 anchor；本例没有 EOS。','<strong>回到生成草稿</strong>新上下文特征 → 新草稿 → 再分预算 → 再验证。']),take:'anchor 已采样出来，但它自身对应的目标模型状态需要在下一轮前向中计算。',math:String.raw`\text{确定前缀}_{t+1}=\text{确定前缀}_t+\text{接受草稿}+\text{纠正或追加 token}`,source:3},
 {title:'最后核对：为什么纠正后仍是目标分布？',text:'回到 A2 的三项概率表。每个词直接被接受的概率质量是 min(p, q)，总计 0.5；余下 0.5 的拒绝质量全部按残差补给“文件”。最后得到 0.7、0.2、0.1，正好回到目标 p。这说的是反复抽样的分布，不保证某次文字与普通解码一样。',view:()=>panel('固定同一前缀，枚举概率质量',table(['词','直接接受质量','拒绝后补回','最终概率 = p'],correctedLaw(verification.A2.p,verification.A2.q).map((v,i)=>row([verification.A2.vocab[i],f(Math.min(verification.A2.p[i],verification.A2.q[i])),f(v-Math.min(verification.A2.p[i],verification.A2.q[i])),f(v)])))) ,take:'调度决定花多少计算；目标 p、草稿 q 和残差规则共同决定采样结果。置信度不能代替验证。',math:String.raw`\min(p(v),q(v))+[p(v)-q(v)]_+=p(v)`,source:2}
];
let current=0,playing=false,timer=null,renderId=0,mathQueue=Promise.resolve();
const starts=[0,5,11,20],hashes=['draft','schedule','verify','output'];
function group(i){return i>=20?3:i>=11?2:i>=5?1:0;}
function tokenProbability(r,j,i){
 const key=`${r.id}${j+1}`,v=verification[key];
 if(i<4)return '';
 if(i<=10){
  const cs=r.confidence.slice(0,j+1),a=survival(r.confidence)[j];
  return `<div class="token-probability"><span class="probability-label">条件接受率估计</span>${tex(`c_{${r.id},${j+1}}=${f(r.confidence[j])}`)}${i>=5?`<div class="token-survival"><span class="probability-label">前缀存活率 · 累乘</span>${tex(`a_{${r.id},${j+1}}=${cs.map((_,k)=>`c_{${k+1}}`).join('')}`)}${j>0?tex('='+cs.map(fix=>f(fix)).join('\\times')):''}${tex(`=${f(a)}`)}</div>`:''}</div>`;
 }
 if(!v)return '<div class="token-probability unverified"><span>未进入草稿验证</span><small>不填验证概率，<br>也不把它记为 0。</small></div>';
 if(i<13)return '<div class="token-probability unverified">已安排目标前向<br><small>下一步展开 p 与 q</small></div>';
 const p=v.p[v.candidate],q=v.q[v.candidate],a=acceptance(v.p,v.q,v.candidate),checked=i>=({A1:14,A2:15,B1:18}[key]);
 return `<div class="token-probability"><span class="probability-label">当前候选的概率</span>${tex(`q=${f(q)},\\quad p=${f(p)}`)}<div class="token-survival"><span class="probability-label">采样接受概率</span>${tex(`\\alpha=\\min\\!\\left(1,\\frac{${f(p)}}{${f(q)}}\\right)`)}${tex(a===1?'=1':Math.abs(a-1/3)<1e-12?'=\\frac{1}{3}':'\\approx '+f(a))}${checked?`<div class="token-verdict ${v.u<a?'pass':'fail'}">${tex(`u=${v.u}${v.u<a?'<':'>'}\\alpha`)}<strong>${v.u<a?'接受':'拒绝'}</strong></div>`:''}</div></div>`;
}
function lanes(i){return requests.map(r=>{
 const detailed=i>=4;
 let done=i>=3?3:i===2&&r.id==='A'?1:0;
 let ts=r.draft.map((word,j)=>{
  let cls='',label=`${r.id}${j+1} · 草稿`;
  if(j>=done)return token('·',`${r.id}${j+1} · 待生成`,'anchor');
  if(i>=10&&((r.id==='A'&&j>=2)||(r.id==='B'&&j>=1))){cls='pruned';label=`${r.id}${j+1} · 未调度`;}
  else if(r.id==='A'&&j===0&&i>=14||r.id==='B'&&j===0&&i>=18){cls='accepted';label=`${r.id}${j+1} · 接受`;}
  else if(r.id==='A'&&j===1&&i>=15){cls='rejected';label='A2 · 拒绝';}
  else if(i>=8&&(r.id==='A'&&(j===0||i>=9&&j===1)||r.id==='B'&&i>=10&&j===0)){label=`${r.id}${j+1} · 已调度`;}
  return detailed?`<div class="token-with-probability ${cls}" data-candidate="${r.id}${j+1}">${token(word,label,cls)}${tokenProbability(r,j,i)}</div>`:token(word,label,cls);
 }).join('');
 let output='';
 if(r.id==='A'&&i>=14)output=token('打开','草稿接受','accepted')+(i>=16?token(corrected,'残差纠正','sampled enter'):'');
 if(r.id==='B'&&i>=18)output=token('你好','草稿接受','accepted')+(i>=19?token(added,'目标追加','sampled enter'):'');
 const explanation=i>=4&&i<=10?`<p class="probability-context">下方 c 是历史条件接受率估计，不是目标词概率 p。${i>=5?'每格中的 c₁、c₂、c₃ 都属于本请求；a 是“从第一枚到这里全部通过”的估计概率。':''}</p>`:i>=13?'<p class="probability-context">p、q 对应相同前缀下的当前候选；α 才是本轮采样接受概率。未调度的草稿不补造验证数值。</p>':'';
 return `<div class="lane ${detailed?'lane-detailed':''}"><div class="lane-header"><strong>请求 ${r.id}</strong><small>prompt：${r.prompt}</small></div><p class="stage-label">本轮输入与草稿（尚不等于交付）</p>${detailed?`<div class="probability-anchor">已确定的 anchor：<strong>${r.anchor}</strong><span>↓ 继续生成下面三枚草稿</span></div><div class="probability-tokens">${ts}</div>`:`<div class="tokens">${token(r.anchor,'旧 anchor','anchor')}<span class="arrow">→</span>${ts}</div>`}${explanation}${output?`<div class="confirmed"><span>已确定新增</span><div class="tokens">${output}</div></div>`:''}</div>`;
 }).join('');}
async function typeset(id){
 try {if(!window.MathJax?.startup?.promise)return;await window.MathJax.startup.promise;if(id!==renderId)return;await window.MathJax.typesetPromise([$('lanes'),$('workbench'),$('formula')]);}
 catch(e){$('error').hidden=false;console.error(e);}
}
function show(i,updateHash=true){
 current=Math.max(0,Math.min(frames.length-1,i));const s=frames[current],g=group(current),id=++renderId;
 window.MathJax?.typesetClear?.([$('lanes'),$('workbench'),$('formula')]);
 $('phase').textContent=`第 ${g+1} 节 · ${['草稿从哪里来','调度怎样计算','验证怎样发生','交付与下一轮'][g]}`;
 $('title').textContent=s.title;$('narration').textContent=s.text;$('lanes').innerHTML=lanes(current);$('workbench').innerHTML=s.view();fitDraftDiagram();$('takeaway').textContent=s.take;$('formula').innerHTML=(s.guide?s.guide():'')+tex(s.math)+(s.example?s.example():'');
 $('count').textContent=`第 ${current+1} / ${frames.length} 页`;$('timeline').value=current;$('timeline').setAttribute('aria-valuetext',`${current+1} / ${frames.length}：${s.title}`);
 $('prev').disabled=current===0;$('next').disabled=current===frames.length-1;
 document.querySelectorAll('[data-page]').forEach((b,j)=>b.setAttribute('aria-current',j===g?'true':'false'));
 const [label,url]=sources[s.source];$('source-line').innerHTML=`本页机制依据：<a href="${url}" target="_blank" rel="noopener">${label}</a> · 数字与措辞为本站教学例。`;
 if(updateHash)history.replaceState(null,'',`#page-${current+1}`);
 mathQueue=mathQueue.then(()=>typeset(id));
 if(current===frames.length-1)pause();
}
function pause(){playing=false;clearTimeout(timer);timer=null;$('play').textContent='自动放映';$('play').setAttribute('aria-pressed','false');$('mode').textContent='手动翻页 · 每次只播放本页';}
function schedule(){clearTimeout(timer);if(!playing)return;const duration=Math.max(9000,frames[current].text.length*110)/Number($('speed').value);timer=setTimeout(()=>{show(current+1);alignCaption();schedule();},duration);}
function alignCaption(){const rect=document.querySelector('.caption').getBoundingClientRect(),offset=document.querySelector('.transport').getBoundingClientRect().height+12;if(rect.top<offset||rect.top>innerHeight*.6)window.scrollTo({top:scrollY+rect.top-offset,behavior:'instant'});}
function move(i){pause();show(i);alignCaption();}
$('restart').onclick=()=>move(0);$('prev').onclick=()=>move(current-1);$('next').onclick=()=>move(current+1);
$('play').onclick=()=>{if(playing){pause();return;}if(current===frames.length-1)show(0);playing=true;$('play').textContent='暂停放映';$('play').setAttribute('aria-pressed','true');$('mode').textContent='自动放映 · 每页留出阅读时间';schedule();};
$('speed').onchange=()=>schedule();$('timeline').oninput=e=>move(Number(e.target.value));
for(const b of document.querySelectorAll('[data-page]'))b.onclick=()=>move(Number(b.dataset.page));
document.addEventListener('keydown',e=>{if(e.target.closest('input,select,button,a,summary,textarea,[contenteditable]'))return;if(e.key==='ArrowRight'){e.preventDefault();move(current+1);}if(e.key==='ArrowLeft'){e.preventDefault();move(current-1);}if(e.code==='Space'){e.preventDefault();$('play').click();}});
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
function fromHash(){const h=location.hash.slice(1),n=hashes.indexOf(h),page=/^page-(\d+)$/.exec(h);if(n>=0)show(starts[n],false);else if(page)show(Number(page[1])-1,false);else if(!h||h==='input')show(0,false);}
window.addEventListener('hashchange',()=>{pause();fromHash();});
$('timeline').max=frames.length-1;show(0,false);fromHash();
window.addEventListener('load',()=>{mathQueue=mathQueue.then(()=>typeset(renderId));});

window.matchMedia('(max-width:700px)').addEventListener('change',fitDraftDiagram);
$('workbench').addEventListener('click',e=>{const button=e.target.closest('.draft-zoom');if(!button)return;const expanded=button.closest('.draft-figure').classList.toggle('is-expanded');button.setAttribute('aria-pressed',String(expanded));button.textContent=expanded?'返回流程全景':'放大完整图';fitDraftDiagram();});
