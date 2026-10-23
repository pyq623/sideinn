/* A deterministic teaching model. These weights are invented, not DeepSeek weights. */
export const TOKENS = ['请','记住','暗号','是','蓝鲸','。','重复','暗号','蓝鲸'];
export const EMBEDDINGS = [[.2,.6],[-.4,.5],[.8,-.2],[.1,-.7],[1,.3],[-.5,-.4],[.4,.9],[.8,-.2],[1,.3]];
export const MODES = ['','SWA','Full','Reuse','Full','Reindex','Reuse'];
export const W = 3;
export const Q = [null,[[.8,.1],[-.1,.9]],[[.9,-.1],[.2,.8]],[[.7,.2],[-.2,1]],[[1,.1],[-.1,.8]],[[.7,-.3],[.3,.9]],[[.9,.2],[-.1,.7]]];
export const INDEX_Q = {2:[[1,0],[0,1]],4:[[1,0],[0,1]],5:[[0,-1],[1,0]]};
export const PROJECTION = {A:[[.8,.2],[-.1,.9]],C:[[.9,-.2],[.1,.8]]};
export const mul = (v,m) => [v[0]*m[0][0]+v[1]*m[1][0],v[0]*m[0][1]+v[1]*m[1][1]];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1];
const top=(rows,k=2)=>[...rows].sort((a,b)=>b.score-a.score||a.pos-b.pos).slice(0,k).map(a=>a.pos);
export const positions=(start,end)=>Array.from({length:Math.max(0,end-start+1)},(_,j)=>start+j);
export const localMatrix=l=>[[.65+.03*l,0],[0,.85-.02*l]];
const empty=()=>({h:Array.from({length:7},()=>Array(10).fill(null)),local:Array.from({length:7},()=>Array(10).fill(null)),traces:Array.from({length:7},()=>Array(10).fill(null)),shared:{A:Array(10).fill(null),C:Array(10).fill(null)},n:8,start:1});
function poolFrom(scores) {
 const blocks=new Map();
 for(const row of scores){const id=Math.floor((row.pos-1)/2);const b=blocks.get(id)||{pos:id,score:-Infinity,items:[]};b.score=Math.max(b.score,row.score);b.items.push(row.pos);blocks.set(id,b);}
 return top([...blocks.values()]).flatMap(id=>blocks.get(id).items).sort((a,b)=>a-b);
}
function processLayer(state,l,queries,{start=1,decode=false,hierarchy=true,keepGlobal=false}={}) {
 const key=l<4?'A':'C';
 if((l===2||l===4)&&!keepGlobal){
  const source=l===2?1:3;
  for(const j of (decode?queries:positions(1,state.n))) state.shared[key][j]=mul(state.h[source][j],PROJECTION[key]);
 }
 for(const i of queries) state.local[l][i]=mul(state.h[l-1][i],localMatrix(l));
 for(const i of queries){
  const input=state.h[l-1][i],q=mul(input,Q[l]);
  const localPositions=positions(Math.max(start,i-W+1),i);
  let indexQ=null,scores=[],selected=[],pool=[],scanned=[];
  if(l>1){
   if(MODES[l]==='Reuse'){
    const previous=state.traces[l-1][i];selected=previous.selected;pool=previous.pool;
   }else{
    indexQ=mul(input,INDEX_Q[l]);
    scanned=decode&&hierarchy&&l===5?state.traces[4][i].pool:positions(1,i);
    scores=scanned.map(pos=>({pos,score:dot(indexQ,state.shared[key][pos])/Math.sqrt(2)}));
    selected=top(scores);
    if(l===4&&decode&&hierarchy) pool=poolFrom(scores);
    if(l===5&&decode&&hierarchy) pool=state.traces[4][i].pool;
   }
  }
  // Global and local entries are separate representations even at the same token position.
  const entries=[...selected.map(pos=>({kind:'global',pos,kv:state.shared[key][pos]})),...localPositions.map(pos=>({kind:'local',pos,kv:state.local[l][pos]}))];
  const logits=entries.map(e=>dot(q,e.kv)/Math.sqrt(2)),max=Math.max(...logits),exp=logits.map(v=>Math.exp(v-max)),sum=exp.reduce((a,b)=>a+b,0);
  const attention=[0,0];entries.forEach((e,j)=>{e.score=logits[j];e.weight=exp[j]/sum;for(let k=0;k<2;k++)attention[k]+=e.weight*e.kv[k];});
  const output=input.map((v,k)=>v+.3*attention[k]);
  state.h[l][i]=output;
  state.traces[l][i]={l,i,input,q,indexQ,scores,selected,pool,scanned,localPositions,entries,attention,output,key:l>1?key:null};
 }
}
export function runFull(n=8){const state=empty();state.n=n;for(const i of positions(1,n))state.h[0][i]=[...EMBEDDINGS[i-1]];for(let l=1;l<=6;l++)processLayer(state,l,positions(1,n));return state;}
export function runBounded(full,replay=3){
 const s=empty();s.start=9-replay;s.shared.A=full.shared.A;s.shared.C=full.shared.C;
 for(let l=0;l<=3;l++){s.h[l]=full.h[l].slice();s.local[l]=full.local[l].slice();s.traces[l]=full.traces[l].slice();}
 for(let l=4;l<=6;l++)processLayer(s,l,positions(s.start,8),{start:s.start,keepGlobal:true});return s;
}
export function restorePrefix(full,replay=3){
 const s=empty();s.start=9-replay;s.shared.A=full.shared.A;s.shared.C=full.shared.C;
 for(const i of positions(s.start,8))s.h[0][i]=[...EMBEDDINGS[i-1]];
 for(let l=1;l<=6;l++)processLayer(s,l,positions(s.start,8),{start:s.start,keepGlobal:true});return s;
}
export function decodeOne(before,{hierarchy=true}={}){
 const s={h:before.h.map(r=>r.slice()),local:before.local.map(r=>r.slice()),traces:before.traces.map(r=>r.slice()),shared:{A:before.shared.A.slice(),C:before.shared.C.slice()},n:9,start:before.start};
 s.h[0][9]=[...EMBEDDINGS[8]];
 for(let l=1;l<=6;l++)processLayer(s,l,[9],{start:1,decode:true,hierarchy,keepGlobal:!(l===2||l===4)});
 return s;
}
export const delta=(a,b)=>Math.max(...a.map((v,i)=>Math.abs(v-b[i])));
