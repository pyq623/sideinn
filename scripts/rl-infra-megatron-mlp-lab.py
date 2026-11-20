#!/usr/bin/env python3
"""CPU-only, standard-library check of a two-shard MLP; no distributed runtime."""
import json
from pathlib import Path

def transpose(a):
    return [list(row) for row in zip(*a)]

def mm(a, b):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]

def add(a, b):
    return [[x+y for x,y in zip(ar,br)] for ar,br in zip(a,b)]

def relu(a):
    return [[max(0,x) for x in row] for row in a]

def derivative(g, u):
    return [[x if y>0 else 0 for x,y in zip(gr,ur)] for gr,ur in zip(g,u)]

def error(a, b):
    return max(abs(x-y) for ar,br in zip(a,b) for x,y in zip(ar,br))

def main():
    x=[[1,2],[2,-1]]
    a=[[1,0,1,-1],[0,1,1,2]]
    b=[[1,0],[0,1],[1,1],[2,-1]]
    u=mm(x,a); y=relu(u); z=mm(y,b)
    # L = 1/2 sum(Z**2), so G = Z. There are no ReLU inputs at zero.
    g=z
    d=derivative(mm(g,transpose(b)),u)
    dx=mm(d,transpose(a)); da=mm(transpose(x),d); db=mm(transpose(y),g)
    shards=[]
    for i in range(2):
        ai=[row[2*i:2*i+2] for row in a]; bi=b[2*i:2*i+2]
        ui=mm(x,ai); yi=relu(ui); zi=mm(yi,bi)
        di=derivative(mm(g,transpose(bi)),ui)
        shards.append(dict(z=zi,dx=mm(di,transpose(ai)),da=mm(transpose(x),di),db=mm(transpose(yi),g)))
    joined_da=[r0+r1 for r0,r1 in zip(shards[0]['da'],shards[1]['da'])]
    joined_db=shards[0]['db']+shards[1]['db']
    errs={'output':error(z,add(shards[0]['z'],shards[1]['z'])),
          'input_gradient':error(dx,add(shards[0]['dx'],shards[1]['dx'])),
          'A_gradient':error(da,joined_da),'B_gradient':error(db,joined_db)}
    assert all(v==0 for v in errs.values())
    wrong={'omit_forward_sum_keep_shard0':error(z,shards[0]['z']),
           'omit_backward_sum_keep_shard0_with_correct_G':error(dx,shards[0]['dx'])}
    assert all(v>0 for v in wrong.values())
    result={'scope':'CPU mathematical decomposition only; no GPU or collective execution',
            'X':x,'A':a,'B':b,'Z':z,'dX':dx,'dA':da,'dB':db,'shards':shards,
            'max_absolute_errors':errs,'intentional_omission_errors':wrong}
    dest=Path(__file__).resolve().parents[1]/'assets/rl-infra-megatron/mlp-check.json'
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    main()
