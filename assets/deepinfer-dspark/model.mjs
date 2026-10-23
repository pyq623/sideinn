// Deterministic teaching data; no real model weights or measured hardware timings.
export const requests = [
 {id:'A',prompt:'请执行操作。',anchor:'先',draft:['打开','测试','。'],confidence:[.9,.9,.9]},
 {id:'B',prompt:'打个招呼。',anchor:'说',draft:['你好','朋友','！'],confidence:[.8,.5,.5]}
];
export const sum = xs => xs.reduce((a,b)=>a+b,0);
export function survival(cs){let a=1;return cs.map(c=>a*=c);}
export function rank(rs=requests){return rs.flatMap((r,ri)=>survival(r.confidence).map((a,j)=>({id:r.id,j:j+1,a,ri}))).sort((a,b)=>b.a-a.a||a.ri-b.ri||a.j-b.j);}
export function allocate(total,rs=requests){if(!Number.isInteger(total)||total<rs.length||total>rs.length+rs.reduce((n,r)=>n+r.confidence.length,0))throw new RangeError('Invalid query budget');const selected=rank(rs).slice(0,total-rs.length);return {selected,lengths:rs.map(r=>selected.filter(s=>s.id===r.id).length),expected:rs.length+sum(selected.map(s=>s.a))};}
// Historical estimates for choosing the total budget, assumed equal to the newer allocation estimates in this example.
export const historicalRequests = requests.map(r=>({...r,confidence:[...r.confidence]}));
export const costs=[1.2,1.4,1.7,2,2.5,2.8,3]; // milliseconds for total query counts 2..8
export const budgets=costs.map((ms,i)=>({total:i+2,ms,...allocate(i+2,historicalRequests)})).map(row=>({...row,tps:1000*row.expected/row.ms}));
export const best=budgets.reduce((a,b)=>b.tps>a.tps?b:a);
export function softmax(xs){const m=Math.max(...xs),es=xs.map(x=>Math.exp(x-m));return es.map(x=>x/sum(es));}
export const markov={base:[.4,.3,.3],afterOpen:[Math.log(.2/.4),Math.log(.6/.3),Math.log(.2/.3)],afterRun:[Math.log(.1/.4),Math.log(.8/.3),Math.log(.1/.3)]};
export const markovQ=softmax(markov.base.map((p,i)=>Math.log(p)+markov.afterOpen[i]));
export function sample(p,u){if(u<0||u>=1)throw new RangeError('u outside [0,1)');let c=0;for(let i=0;i<p.length;i++){c+=p[i];if(u<c)return i;}return p.length-1;}
export function acceptance(p,q,i){if(q[i]<=0)throw new RangeError('Draft token must have positive q');return Math.min(1,p[i]/q[i]);}
export function residual(p,q){const mass=p.map((v,i)=>Math.max(v-q[i],0)),z=sum(mass);return z>0?mass.map(v=>v/z):null;}
export const verification={
 A1:{vocab:['打开','运行','跳过'],q:[.6,.3,.1],p:[.8,.1,.1],candidate:0,u:.4,draftU:.2,context:'A 的旧前缀 + 先'},
 A2:{vocab:['文件','测试','目录'],q:markovQ,p:[.7,.2,.1],candidate:1,u:.8,draftU:.55,context:'A 的旧前缀 + 先 + 打开'},
 B1:{vocab:['你好','再见','谢谢'],q:[.5,.3,.2],p:[.6,.2,.2],candidate:0,u:.4,draftU:.2,context:'B 的旧前缀 + 说'}
};
export const correction=residual(verification.A2.p,verification.A2.q);
export const bonus={vocab:['朋友','。','！'],p:[.1,.8,.1],u:.5};
export const corrected=verification.A2.vocab[sample(correction,.3)];
export const added=bonus.vocab[sample(bonus.p,bonus.u)];
// Enumerate probability mass, rather than trusting a finite Monte Carlo sample.
export function correctedLaw(p,q){const accepted=p.map((v,i)=>Math.min(v,q[i])),reject=1-sum(accepted),r=residual(p,q);return accepted.map((v,i)=>v+reject*(r?.[i]||0));}
