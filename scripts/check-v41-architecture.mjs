// Run from any directory: node scripts/check-v41-architecture.mjs
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {steps,regions} from '../assets/v41-architecture/steps.mjs';
assert.equal(steps.length,22);
assert.equal(new Set(steps.map(s=>s.id)).size,steps.length);
// Exclude annotation-only repetitions when unfolding the actual forward path.
const forward=steps.filter(s=>!['engram','candidate','decode'].includes(s.id));
assert.deepEqual(forward.flatMap(s=>s.layers),Array.from({length:40},(_,i)=>i));
for(const step of steps){
 for(const id of step.focus)assert.ok(regions[id],`Unknown region ${id}`);
 assert.ok(step.layers.every(x=>Number.isInteger(x)&&x>=0&&x<40));
 assert.ok(step.cache[0]>=0&&step.cache[0]<=3&&step.cache[1]>=0&&step.cache[1]<=1);
 assert.ok(step.paths.every(x=>['hidden','cache','index','memory'].includes(x.kind)));
}
const byId=id=>steps.find(s=>s.id===id);
assert.deepEqual(byId('encoder-full').cache,[1,0]);
assert.deepEqual(byId('encoder-group2').cache,[2,0]);
assert.deepEqual(byId('encoder-group3').cache,[3,0]);
assert.deepEqual(byId('global-path').cache,[3,1]);
for(const step of steps.slice(9))assert.deepEqual(step.cache,[3,1]);
assert.equal(byId('global-path').paths[0].kind,'cache');
assert.equal(byId('tail-path').paths[0].kind,'hidden');
assert.deepEqual(byId('reindex').layers,[24]);
assert.deepEqual(byId('decoder-group4').layers,[36,37,38,39]);
const manifest=JSON.parse(readFileSync(new URL('../assets/v41-flash-figures/official-sources.json',import.meta.url)));
const original=manifest.find(x=>x.file==='official-fig3-architecture.svg');
assert.equal(createHash('sha256').update(readFileSync(new URL('../assets/v41-flash-figures/'+original.file,import.meta.url))).digest('hex'),original.sha256);
console.log('PASS: 22 steps; exact 0–39 forward coverage; correct KV ownership; distinct CED paths; original figure unchanged.');
