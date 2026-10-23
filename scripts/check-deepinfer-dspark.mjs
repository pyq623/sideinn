import assert from 'node:assert/strict';
import {requests,survival,allocate,best,budgets,verification,acceptance,residual,correctedLaw,sample,markovQ,corrected,added} from '../assets/deepinfer-dspark/model.mjs';
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
assert.equal(best.total,5);close(best.expected,4.51);close(best.tps,2255);
assert.deepEqual(allocate(5).lengths,[2,1]);assert.deepEqual(allocate(5).selected.map(r=>r.id+r.j),['A1','A2','B1']);
for(let total=2;total<=8;total++){
 const a=allocate(total);assert.equal(a.lengths.reduce((x,y)=>x+y,2),total);
 // Exhaustively compare every valid prefix allocation with this budget.
 let optimum=-Infinity;
 for(let x=0;x<=3;x++)for(let y=0;y<=3;y++)if(2+x+y===total){const score=2+survival(requests[0].confidence).slice(0,x).reduce((a,b)=>a+b,0)+survival(requests[1].confidence).slice(0,y).reduce((a,b)=>a+b,0);optimum=Math.max(score,optimum);}
 close(a.expected,optimum);
 for(const r of requests){const pos=a.selected.filter(s=>s.id===r.id).map(s=>s.j).sort((a,b)=>a-b);assert.deepEqual(pos,Array.from({length:pos.length},(_,i)=>i+1));}
}
for(const v of Object.values(verification)){
 close(v.q.reduce((a,b)=>a+b),1);close(v.p.reduce((a,b)=>a+b),1);
 assert.equal(sample(v.q,v.draftU),v.candidate);
 correctedLaw(v.p,v.q).forEach((p,i)=>close(p,v.p[i]));
}
assert.ok(verification.A1.u<acceptance(verification.A1.p,verification.A1.q,0));
assert.ok(verification.A2.u>acceptance(verification.A2.p,verification.A2.q,1));
assert.ok(verification.B1.u<acceptance(verification.B1.p,verification.B1.q,0));
markovQ.forEach((v,i)=>close(v,[.2,.6,.2][i]));assert.equal(corrected,'文件');assert.equal(added,'。');
// Invariance over many different probability tables, including equal and disjoint supports.
for(const [p,q] of [[[.2,.8],[.2,.8]],[[1,0],[0,1]],[[0,.3,.7],[.6,.3,.1]]])correctedLaw(p,q).forEach((v,i)=>close(v,p[i]));
assert.equal(residual([.2,.8],[.2,.8]),null);
close(budgets.at(-1).expected,5.839);
console.log('PASS: budget optimum, exhaustive prefix allocation, sampled drafts, accept/reject decisions, Markov correction, residual normalization and exact target probability mass.');
