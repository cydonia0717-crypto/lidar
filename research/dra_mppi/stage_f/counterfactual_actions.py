"""Evaluator-only bounded action-library replay, not a deployable controller.

Uses known future truth to test existence of physical avoidance in a restricted
library. Failure to find a path is NOT a proof that no safe control exists.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
from analyze import arc, truth


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();snapshots=[]
    for line in a.input.open():
        r=json.loads(line)
        if r['episode']=='turn_101_event_cdf_cv' and r['time'] in [0.,1.2,1.6,2.0]:snapshots.append(r)
    actions=list(itertools.product([0.,.5,1.,1.5,2.],[-1.,-.5,0.,.5,1.]))
    backups=[(0.,0.),(1.,-1.),(1.,1.),(2.,-1.),(2.,1.)]
    results=[]
    for r in snapshots:
        now=r['time'];anchor=r['truth_at_end'][0][0]+.9*max(0,now+.2-1.4)
        ts=[k*.01 for k in range(401)]
        targets=[truth('turn',anchor,1,now+t) for t in ts]
        best=-math.inf;choice=None;safe=0
        for u,b,duration in itertools.product(actions,backups,[.2,.6,1.]):
            q0=arc(r['state'],u,duration)
            junction=[q0[0],q0[1],r['state'][2]+u[1]*duration]
            mingap=math.inf
            for tau,obstacles in zip(ts,targets):
                q=arc(r['state'],u,tau) if tau<=duration else arc(junction,b,tau-duration)
                gap=min(min(math.dist(q,o)-.625 for o in obstacles),3.5-abs(q[1])-.325)
                mingap=min(mingap,gap)
                if mingap-.019<=best:break
            lower=mingap-.019
            # A pruned candidate cannot improve the incumbent; safe-count is
            # deliberately not reported because pruning skips its full trace.
            if lower>best:best=lower;choice={'prefix':u,'duration':duration,'backup':b}
        results.append(dict(time=now,original_next_risk=r['cdf_next'],sampled_model_safe=r['sample_safe_count'],
            best_physical_gap_lower_m=best,choice=choice,
            scope='375 bounded two-phase sequences;4s/10ms with0.019m between-node bound;future truth diagnostic;not Gaussian risk feasibility or global reachability'))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))


if __name__=='__main__':main()
