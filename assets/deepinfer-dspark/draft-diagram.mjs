// Original vector teaching diagram, adapted from the layout of DSpark Figure 1.
// Three positions match this walkthrough (the paper illustration uses four).
export function draftDiagram(stage='parallel') {
 const first=stage!=='parallel',complete=stage==='complete',xs=[480,665,850];
 const label=(x,y,s,cls='')=>`<text x="${x}" y="${y}" class="${cls}" text-anchor="middle">${s}</text>`;
 const rect=(x,y,w,h,cls,rx=12)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" class="${cls}"/>`;
 const arrow=(d,cls='')=>`<path d="${d}" class="draft-arrow ${cls}" marker-end="url(#draft-arrowhead)"/>`;
 const math=(x,y,i)=>`<foreignObject x="${x-65}" y="${y}" width="130" height="45"><div xmlns="http://www.w3.org/1999/xhtml" class="draft-math">\\(U_${i}\\)</div></foreignObject>`;
 return `<figure class="draft-figure" aria-label="DSpark 并行骨干与轻量顺序头的数据流">
 <div class="draft-figure-heading"><div><span>请求 A · 草稿生成结构</span><h3>${complete?'同一张图：顺序采样已完成':first?'同一张图：第一枚草稿接入下一位置':'沿着箭头，从下向上读'}</h3></div><span class="draft-stage-badge">${complete?'三个候选已生成':first?'当前：轻量顺序头':'当前：并行骨干'}</span></div>
 <p class="draft-mobile-context">上一轮目标模型提供 <strong>anchor「先」</strong> 和上下文特征；下图聚焦右侧草稿流程。</p><button class="draft-zoom" aria-pressed="false">放大完整图</button>
 <div class="draft-scroll" tabindex="0" role="region" aria-label="草稿结构图；窄屏可横向滚动">
 <svg class="draft-diagram" viewBox="0 0 1000 720" role="img" aria-labelledby="draft-title draft-desc">
 <title id="draft-title">${stage==='parallel'?'anchor 与两个 mask 同时经过并行骨干，生成三个位置的基础 logits':'基础 logits 经过顺序头，从左到右采样草稿'}</title>
 <desc id="draft-desc">左侧是已完成的上一轮目标模型，提供 anchor“先”和上下文特征。右侧下方是三个输入位，中间是 Parallel Block，上方是三个 logits，再上方是 Sequential Block。并行阶段三个位置同时计算；顺序阶段前一枚采样结果影响后一位置。</desc>
 <defs><marker id="draft-arrowhead" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 1 1 L 9 5 L 1 9 Z" fill="context-stroke"/></marker></defs>
 ${rect(20,285,250,355,'draft-previous',24)}
 ${label(145,316,'上一轮 · 已完成','draft-small')}
 ${rect(80,345,130,58,'draft-anchor')}${label(145,373,'先','draft-token-word')}
 ${label(145,395,'anchor','draft-tiny')}
 ${arrow('M145 458V410','draft-known')}
 ${rect(43,458,204,82,'draft-target',32)}
 ${label(145,491,'Target Model','draft-block-title')}${label(145,519,'目标模型','draft-subtitle')}
 ${arrow('M145 578V546','draft-known')}
 ${rect(48,581,194,38,'draft-prefix',8)}${label(145,606,'更早的确定前缀','draft-small')}
 ${rect(322,20,650,665,'draft-generation',26)}
 ${label(647,55,'DSpark · 本例生成 3 枚草稿','draft-block-title')}
 <g class="${complete?'draft-ready':'draft-future'}">
 ${rect(367,79,560,48,'draft-next',12)}${label(647,109,'下一阶段：分配验证名额 → 目标模型验证','draft-subtitle')}
 ${xs.map(x=>arrow(`M${x} 177V136`,'draft-dashed')).join('')}
 </g>
 ${xs.map((x,i)=>{const ready=complete||first&&i===0;return `<g class="${ready?'draft-sampled':'draft-future'}">${rect(x-65,180,130,58,ready?'draft-candidate':'draft-pending')}${label(x,204,`A${i+1}`,'draft-tiny')}${label(x,227,ready?['打开','测试','。'][i]:'待采样','draft-output-word')}</g>`;}).join('')}
 <g class="${first?'draft-sequential-active':'draft-future'}">
 ${xs.map((x,i)=>arrow(`M${x} 274V244`,complete||first&&i===0?'draft-live':'draft-future')).join('')}
 ${rect(355,278,580,58,'draft-sequential',29)}
 ${label(645,303,'Sequential Block · 轻量顺序头','draft-block-title')}
 ${label(645,325,'从左到右采样，前一个词影响下一个词','draft-small')}
 ${arrow('M545 209C580 209 563 360 602 360H665V342','draft-dependency '+(first?'draft-live':'draft-future'))}
 ${arrow('M730 209C765 209 748 360 787 360H850V342','draft-dependency '+(complete?'draft-live':'draft-future'))}
 </g>
 ${xs.map(x=>arrow(`M${x} 378V342`,first?'draft-live':'draft-future')).join('')}
 ${xs.map((x,i)=>`<g class="draft-logit"><rect x="${x-65}" y="382" width="130" height="55" rx="10" class="draft-logit-box"/>${math(x,382,i+1)}${label(x,431,'基础 logits','draft-tiny')}</g>`).join('')}
 ${xs.map(x=>arrow(`M${x} 476V443`,'draft-parallel-flow')).join('')}
 <g class="draft-backbone">
 ${rect(355,480,580,84,'draft-parallel',40)}
 ${label(645,515,'Parallel Block · 并行骨干','draft-block-title')}
 ${label(645,543,'一次前向，同时准备三个位置的表示','draft-subtitle')}
 </g>
 ${arrow('M248 495H347','draft-context')}
 ${label(299,465,'上下文','draft-tiny')}${label(299,483,'特征','draft-tiny')}
 ${xs.map(x=>arrow(`M${x} 590V570`,'draft-input-flow')).join('')}
 ${xs.map((x,i)=>`<g class="draft-input">${rect(x-65,595,130,48,i===0?'draft-anchor':'draft-mask')}${label(x,625,i===0?'先':'[mask]','draft-token-word')}${label(x,674,`输入位置 ${i+1}`,'draft-small')}</g>`).join('')}
 ${arrow('M210 374H287V703H480V649','draft-anchor-route')}
 ${label(160,692,'anchor 送入第 1 个位置','draft-small')}
 </svg></div>
 <div class="draft-reading-key"><span><i class="key-parallel"></i>绿色模块：草稿计算</span><span><i class="key-logits"></i>紫色框：logits／候选</span><span>灰色虚线：下一步，尚未执行</span></div>
 <p class="draft-explanation">${complete?'“打开”影响第二位置，随后“测试”影响第三位置。三个草稿仍需经过后续调度和目标验证。':first?'第一位置抽到“打开”，沿弯曲箭头传给第二位置；第二位置把这项依赖加到自己的基础 logits 上。':'三条向上的箭头同时点亮：生成的是三组基础 logits，尚未抽出“打开、测试、。”。上方顺序头将在下一页接着运行。'}</p>
 <figcaption>参考 <a href="https://arxiv.org/html/2607.05147v1#S3.F1" target="_blank" rel="noopener">DSpark Figure 1</a> 的结构重绘，缩为本例的 3 个位置；不是论文原图。<span class="draft-scroll-hint">窄屏可左右滑动查看完整图。</span></figcaption>
 </figure>`;
}

export function fitDraftDiagram(){
 const figure=document.querySelector('.draft-figure');if(!figure)return;
 const mobile=window.matchMedia('(max-width:700px)').matches,expanded=figure.classList.contains('is-expanded');
 figure.querySelector('svg.draft-diagram').setAttribute('viewBox',mobile&&!expanded?'310 10 675 690':'0 0 1000 720');
}
