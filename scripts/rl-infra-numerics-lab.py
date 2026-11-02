#!/usr/bin/env python3
"""CPU teaching experiments, Python standard library only.
Not a simulation of the full NVFP4 encoding or GPU kernels.
Run: python3 scripts/rl-infra-numerics-lab.py
Outputs reproducible JSON and an original SVG into assets/rl-infra-numerics/.
"""
import json
import math
import random
import struct
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'assets' / 'rl-infra-numerics'
SEED = 20261029


def softmax(z):
    e = [math.exp(v-max(z)) for v in z]
    return [v/sum(e) for v in e]


def percentile(values, q):
    values = sorted(values)
    pos = (len(values)-1)*q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo]+(pos-lo)*(values[hi]-values[lo])


def norm(v):
    return math.sqrt(sum(x*x for x in v))


def distance(a, b):
    return norm([x-y for x,y in zip(a,b)])


def angle(a, b):
    return math.degrees(math.acos(max(-1., min(1., sum(x*y for x,y in zip(a,b))/(norm(a)*norm(b))))))


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def quantize(values, scale):
    # Symmetric signed INT4 teaching quantizer, nearest-even rounding.
    return [scale*max(-7,min(7,round(x/scale))) for x in values]


def exact_gradient_experiment():
    p, q, rewards = [.6,.3,.1], [.2,.3,.5], [1.,-.5,.2]
    baseline = sum(x*y for x,y in zip(p,rewards))
    score = [[float(i==j)-p[j] for j in range(3)] for i in range(3)]
    target = [p[j]*(rewards[j]-baseline) for j in range(3)]
    ratios = [x/y for x,y in zip(p,q)]
    rules = {'uncorrected':[1.]*3, 'exact_is':ratios,
             'tis_cap_2':[min(x,2.) for x in ratios],
             'ice_interval_0.5_2':[x if .5<=x<=2. else 0. for x in ratios]}
    rows = {}
    for name,weights in rules.items():
        g = [sum(q[i]*weights[i]*(rewards[i]-baseline)*score[i][j] for i in range(3)) for j in range(3)]
        rows[name] = {'gradient':g,'l2_error':distance(g,target),'angle_degrees':angle(g,target),
                      'ess_fraction':sum(q[i]*weights[i] for i in range(3))**2/sum(q[i]*weights[i]**2 for i in range(3))}
    logits = [math.log(x) for x in p]
    eps=1e-5
    numeric=[]
    for j in range(3):
        a,b=logits.copy(),logits.copy();a[j]+=eps;b[j]-=eps
        numeric.append((sum(x*r for x,r in zip(softmax(a),rewards))-sum(x*r for x,r in zip(softmax(b),rewards)))/(2*eps))
    assert distance(numeric,target)<1e-9
    assert rows['exact_is']['l2_error']<1e-12
    assert rows['tis_cap_2']['l2_error']>.01
    # Fixed-prefix identity does not repair which prefix was sampled.
    # p(h=1)=.8, q(h=1)=.2 and conditional action distributions identical.
    prefix={'target_probability_h1':.8,'behavior_probability_h1':.2,
            'token_only_weight':1.,'token_only_expectation_h1':.2,'full_prefix_corrected_expectation_h1':.8}
    return {'p':p,'q':q,'reward':rewards,'baseline':baseline,'target_gradient':target,'methods':rows,'prefix_counterexample':prefix}


def moe_experiment():
    rng=random.Random(SEED)
    # Three experts, each maps a 2D input to 3 vocabulary logits.
    experts=[[[1.2,-.4],[-.3,.8],[.1,.2]],
             [[-.8,.3],[.7,-.5],[.2,.9]],
             [[.4,.5],[.1,-.2],[-.6,.7]]]
    def logits(x, route):
        return [sum(a*b for a,b in zip(row,x)) for row in experts[route]]
    gaps={name:[] for name in ['natural','route_replay','version_shift']}
    flips=0
    pairs=[]
    for _ in range(512):
        x=[rng.uniform(-1,1),rng.uniform(-1,1)]
        margin=rng.uniform(-.06,.06)
        # Gate perturbation deliberately probes a top-1 boundary.
        scores=[0.,margin,-.8]; scores_q=[0.,margin+.025,-.8]
        ip=max(range(3),key=lambda j:scores[j]);iq=max(range(3),key=lambda j:scores_q[j])
        flips+=ip!=iq
        xq=quantize(x,max(max(abs(v) for v in x)/7,1e-12))
        zq=logits(xq,iq); zq=[round(v/.125)*.125 for v in zq]
        q=softmax(zq);p=softmax(logits(x,ip));replay=softmax(logits(x,iq))
        pnew=softmax([v+b for v,b in zip(logits(x,ip),[.2,-.15,.05])])
        u=rng.random();a=0
        while a<2 and u>sum(q[:a+1]):a+=1
        for name,dist in [('natural',p),('route_replay',replay),('version_shift',pnew)]:
            gaps[name].append(math.log(dist[a])-math.log(q[a]))
        pairs.append([math.log(pnew[a])-math.log(p[a]),gaps['natural'][-1],gaps['version_shift'][-1]])
    assert max(abs(stale+num-total) for stale,num,total in pairs)<1e-12
    result={}
    for name,vals in gaps.items():
        av=[abs(v) for v in vals];rat=[math.exp(v) for v in vals]
        result[name]={'mean_abs_logprob_gap':sum(av)/len(av),'p99_abs_logprob_gap':percentile(av,.99),
                      'max_abs_logprob_gap':max(av),'p99_ratio':percentile(rat,.99),
                      'ratio_outside_0.8_1.2':sum(not .8<=x<=1.2 for x in rat)/len(rat)}
    assert result['route_replay']['mean_abs_logprob_gap']>0
    assert result['route_replay']['mean_abs_logprob_gap']<result['natural']['mean_abs_logprob_gap']
    return {'seed':SEED,'tokens':512,'routing_flip_fraction':flips/512,'measurements':result,
            'note':'Deliberately near-boundary top-1 toy MoE; INT4-like input rounding plus uniform output rounding, NOT NVFP4 or an LLM.'}


def run():
    left=f32(f32(f32(1e8)+f32(-1e8))+f32(1.))
    right=f32(f32(1e8)+f32(f32(-1e8)+f32(1.)))
    assert left==1. and right==0.
    single=quantize([1.,.5],1/7);mixed=quantize([1.,.5],20/7)
    assert single!=mixed
    w,scale,target=.43,.25,1.
    q=lambda x:scale*round(x/scale)
    loss=lambda x:.5*(q(x)-target)**2
    fd=(loss(w+1e-6)-loss(w-1e-6))/(2e-6)
    ste=q(w)-target
    assert fd==0 and ste==-0.5
    result={'schema_version':1,'seed':SEED,'checks_passed':True,
            'float32_reduction':{'left_associative':left,'right_associative':right},
            'batch_scale':{'target':[1.,.5],'alone_or_row_scaled':single,'with_outlier_tensor_scaled':mixed},
            'ste':{'weight':w,'quantized_weight':q(w),'finite_difference':fd,'identity_ste_gradient':ste},
            'importance_sampling':exact_gradient_experiment(),'moe':moe_experiment()}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    # An original plot using the actual output; no external drawing dependency.
    m=result['moe']['measurements'];maxv=max(x['p99_abs_logprob_gap'] for x in m.values())*1.15
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="420" viewBox="0 0 960 420" role="img" aria-label="Toy MoE logprob gap results"><rect width="960" height="420" fill="#faf8f1"/><g font-family="system-ui,sans-serif" fill="#243f34"><text x="36" y="46" font-size="25" font-weight="700">CPU toy MoE: routing replay leaves numerical error</text><text x="36" y="77" font-size="15">512 sampled tokens · seed 20261029 · teaching experiment, not NVFP4 / GPU</text>']
    for i,(name,d) in enumerate(m.items()):
        y=125+i*75
        svg.append(f'<text x="36" y="{y+17}" font-size="17">{name}</text>')
        for j,(k,c) in enumerate([('mean_abs_logprob_gap','#43745e'),('p99_abs_logprob_gap','#bb9355')]):
            val=d[k];width=val/maxv*530
            svg.append(f'<rect x="220" y="{y+j*23}" width="{width:.3f}" height="17" rx="3" fill="{c}"/><text x="{230+width:.3f}" y="{y+j*23+14}" font-size="14">{val:.5f}</text>')
    svg.append('<rect x="36" y="366" width="16" height="16" fill="#43745e"/><text x="60" y="380" font-size="15">Mean |log p − log q|</text><rect x="350" y="366" width="16" height="16" fill="#bb9355"/><text x="374" y="380" font-size="15">P99 |log p − log q| (nats)</text></g></svg>')
    (OUT/'moe-results.svg').write_text(''.join(svg))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    run()
