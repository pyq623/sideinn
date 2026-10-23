import assert from 'node:assert/strict';
import {EMBEDDINGS,runFull,runBounded,restorePrefix,decodeOne,delta} from '../assets/deepinfer-walkthrough/model.mjs';
const full=runFull(),short=runBounded(full,3),all=runBounded(full,8);
for(let l=1;l<=6;l++)for(let i=1;i<=8;i++)assert.deepEqual(all.h[l][i],full.h[l][i],'Full-length replay must equal the reference');
assert.deepEqual(short.h[4][8],full.h[4][8],'First decoder layer, last position has its complete local window');
assert.ok(delta(short.h[5][8],full.h[5][8])>1e-4,'Truncation must propagate through lower-layer local states');
assert.deepEqual(runBounded(full,5).h[5][8],full.h[5][8]);
assert.ok(delta(runBounded(full,5).h[6][8],full.h[6][8])>1e-6);
const snapshot=JSON.stringify(full.shared),restored=restorePrefix(full,3);
assert.strictEqual(restored.shared.A,full.shared.A);assert.strictEqual(restored.shared.C,full.shared.C);
assert.equal(JSON.stringify(full.shared),snapshot,'Restoring local states must not overwrite hit global entries');
assert.ok(delta(restored.h[3][8],full.h[3][8])>1e-4);
assert.deepEqual(restorePrefix(full,8).h[6][8],full.h[6][8]);
const incremental=decodeOne(full,{hierarchy:false}),nine=runFull(9);
for(let l=1;l<=6;l++)assert.ok(delta(incremental.h[l][9],nine.h[l][9])<1e-12,'Cached one-token decode must match untruncated causal recomputation when indexing policy is identical');
const decoded=decodeOne(short);
for(const key of ['A','C']){assert.deepEqual(decoded.shared[key].slice(1,9),short.shared[key].slice(1,9));assert.ok(decoded.shared[key][9]);assert.equal(short.shared[key][9],null);}
assert.ok(decoded.traces[4][9].pool.length<=4);
assert.deepEqual(decoded.traces[5][9].scanned,decoded.traces[4][9].pool);
assert.strictEqual(decoded.traces[6][9].selected,decoded.traces[5][9].selected);
assert.equal(decoded.traces[6][9].scores.length,0);
for(const s of [full,short,restored,decoded])for(const traces of s.traces)for(const t of traces){if(!t)continue;assert.ok(t.entries.every(e=>e.pos<=t.i),'Never attend to future positions');assert.ok(Math.abs(t.entries.reduce((n,e)=>n+e.weight,0)-1)<1e-12);assert.ok(t.output.every(Number.isFinite));if(t.i===9)assert.deepEqual(t.localPositions,[7,8,9]);}
const original=EMBEDDINGS[7];EMBEDDINGS[7]=[100,-50];const perturbed=runFull();EMBEDDINGS[7]=original;
for(let l=1;l<=6;l++)for(let i=1;i<=7;i++)assert.deepEqual(perturbed.h[l][i],full.h[l][i],'A future-token change must not affect earlier outputs');
console.log('PASS: full replay, dependency propagation, cache preservation, incremental decode, cross-layer reuse, causal masks and normalized attention.');
