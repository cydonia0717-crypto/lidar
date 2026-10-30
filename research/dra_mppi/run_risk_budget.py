"""Frozen risk-estimator experiment; all inputs and promotion gates predeclared."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from scipy.optimize import brentq
import torch
from crosscheck_risk import quadrature
from event_risk import exact_isotropic_union, fixed_event_mc, sequential_event_mc, self_check
from run_upstream_audit import load_module, make_obstacles, save_csv, UPSTREAM_SHA


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(a.protocol.read_text())
    (a.output/'frozen_protocol.json').write_text(json.dumps(protocol,indent=2))
    (a.output/'self_check.json').write_text(json.dumps(self_check(),indent=2))
    torch.set_num_threads(2)
    source=a.upstream/'src/mppi_planner/src/prob_mppi/dynamic_obstacles.py'
    cls=load_module(source,'original_risk_budget').DynamicObstacles
    fixtures=[]
    for split, settings in protocol['fixture_splits'].items():
        for n in settings['counts']:
            for sigma in settings['sigmas']:
                for spread in [.35,1.2]:
                    generator=np.random.default_rng(settings['geometry_seed']+n+int(sigma*100)+int(spread*100))
                    angles=np.arange(n)*2*np.pi/n + generator.uniform(-.2,.2,size=n)
                    means=np.stack((spread*np.cos(angles),spread*np.sin(angles)),-1)
                    edge=int(np.argmax(means[:,0]))
                    anchor=means[edge]
                    queries=[]
                    for level in protocol['reference_levels']:
                        def f(offset):
                            q=np.array([[anchor[0]+offset,anchor[1]]])
                            return exact_isotropic_union(means,q,sigma,protocol['radius'])[0]-level
                        offset=brentq(f,0,20,xtol=1e-12)
                        queries.append([anchor[0]+offset,anchor[1]])
                    queries=np.asarray(queries)
                    reference=exact_isotropic_union(means,queries,sigma,protocol['radius'])
                    assert np.max(abs(reference-protocol['reference_levels']))<1e-9
                    density=[]
                    for q,truth in zip(queries,reference):
                        event_quad,density_quad=quadrature(means,q,sigma,protocol['radius'])
                        assert abs(event_quad-truth)<1e-7
                        density.append(density_quad)
                    fixtures.append(dict(id='%s_n%d_s%.2f_r%.2f'%(split,n,sigma,spread),split=split,
                                         sigma=sigma,means=means,queries=queries,reference=reference,
                                         density_reference=np.array(density)))
    (a.output/'fixtures.json').write_text(json.dumps([{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in f.items()} for f in fixtures],indent=2))
    rows,batches=[],[]
    for fixture_index,f in enumerate(fixtures):
        mu,q,ref=f['means'],f['queries'],f['reference']
        for repeat in range(protocol['repeats']):
            seed=protocol['sampling_seed']+1000*fixture_index+repeat
            jobs=[]
            for budget in protocol['budgets']:
                jobs.append(('original_%d'%budget,'original',budget))
                jobs.append(('event_point_%d'%budget,'point',budget))
            jobs += [('fixed_cp_2048','cp',2048),('fixed_cp_20000','cp',20000),('sequential_cp','sequential',20000),('analytic_cdf','exact',0)]
            if repeat%2:jobs.reverse()
            for method,kind,n in jobs:
                if kind=='original':
                    torch.manual_seed(seed)
                    start=time.perf_counter()
                    obj,cfg=make_obstacles(cls,torch.tensor(mu[None],dtype=torch.float32),f['sigma'],len(q),n)
                    obj.integral_radius=cfg['obstacles']['integral_radius']=protocol['radius']
                    est=obj.integrate_one_shot_monte_carlo_circles(torch.tensor(q[:,0:1],dtype=torch.float32),torch.tensor(q[:,1:2],dtype=torch.float32)).numpy()[:,0]
                    result=dict(estimate=est,decision=np.where(est<=protocol['threshold'],'safe','unsafe'),
                                lower=np.full(len(q),np.nan),upper=np.full(len(q),np.nan),samples=np.full(len(q),n),
                                elapsed_s=time.perf_counter()-start,pair_evaluations=None,target_vectors_drawn=None)
                elif kind=='exact':
                    start=time.perf_counter()
                    est=exact_isotropic_union(mu,q,f['sigma'],protocol['radius'])
                    result=dict(estimate=est,decision=np.where(est<=protocol['threshold'],'safe','unsafe'),lower=est,upper=est,
                                samples=np.zeros(len(q)),elapsed_s=time.perf_counter()-start,pair_evaluations=0,target_vectors_drawn=0)
                elif kind=='sequential':
                    result=sequential_event_mc(mu,q,f['sigma'],protocol['radius'],protocol['budgets'],seed,
                                               protocol['threshold'],protocol['family_alpha'])
                else:
                    result=fixed_event_mc(mu,q,f['sigma'],protocol['radius'],n,seed,protocol['threshold'],protocol['family_alpha'],guard=kind=='cp')
                # Fixed abstract candidate utilities test post-selection within the
                # certified bank; these are not a navigation trajectory objective.
                utility=np.array([3,1,4,0,2,7,5,6])
                safe=np.flatnonzero(result['decision']=='safe')
                selected=int(safe[np.argmin(utility[safe])]) if len(safe) else None
                false_safe=(result['decision']=='safe')&(ref>protocol['threshold'])
                false_unsafe=(result['decision']=='unsafe')&(ref<=protocol['threshold'])
                interval_miss=bool(((ref<result['lower'])|(ref>result['upper'])).any()) if kind in ['cp','sequential'] else None
                batches.append(dict(fixture=f['id'],split=f['split'],repeat=repeat,method=method,
                                    elapsed_s=result['elapsed_s'],pair_evaluations=result['pair_evaluations'],target_vectors_drawn=result['target_vectors_drawn'],
                                    safe_count=int((result['decision']=='safe').sum()),unsafe_count=int((result['decision']=='unsafe').sum()),
                                    unresolved_count=int((result['decision']=='unresolved').sum()),false_safe=int(false_safe.sum()),false_unsafe=int(false_unsafe.sum()),
                                    family_interval_miss=interval_miss,selected_query=selected,
                                    selected_false_safe=bool(selected is not None and ref[selected]>protocol['threshold'])))
                for j in range(len(q)):
                    lo,hi=result['lower'][j],result['upper'][j]
                    rows.append(dict(fixture=f['id'],split=f['split'],repeat=repeat,method=method,query=j,
                                     reference_probability=float(ref[j]),source_density_reference=float(f['density_reference'][j]),
                                     estimate=float(result['estimate'][j]),lower=float(lo) if np.isfinite(lo) else None,upper=float(hi) if np.isfinite(hi) else None,
                                     decision=str(result['decision'][j]),samples=int(result['samples'][j]),false_safe=bool(false_safe[j]),false_unsafe=bool(false_unsafe[j])))
        print('RISK_BANK '+f['id'],flush=True)
    save_csv(a.output/'risk_queries.csv',rows)
    save_csv(a.output/'risk_batches.csv',batches)
    summaries=[]
    for split in protocol['fixture_splits']:
        for method in sorted({b['method'] for b in batches}):
            rr=[r for r in rows if r['split']==split and r['method']==method]
            bb=[b for b in batches if b['split']==split and b['method']==method]
            summaries.append(dict(split=split,method=method,queries=len(rr),batches=len(bb),
                                  false_safe=sum(r['false_safe'] for r in rr),false_unsafe=sum(r['false_unsafe'] for r in rr),
                                  unresolved=sum(r['decision']=='unresolved' for r in rr),
                                  mean_absolute_error=float(np.mean([abs(r['estimate']-r['reference_probability']) for r in rr])),
                                  out_of_probability_range=sum(not 0<=r['estimate']<=1 for r in rr),
                                  mean_batch_ms=1000*np.mean([b['elapsed_s'] for b in bb]),p95_batch_ms=1000*np.percentile([b['elapsed_s'] for b in bb],95),
                                  pair_evaluations=sum(b['pair_evaluations'] or 0 for b in bb),
                                  family_interval_misses=sum(b['family_interval_miss'] or 0 for b in bb),
                                  selected_false_safe=sum(b['selected_false_safe'] for b in bb)))
    save_csv(a.output/'risk_summary.csv',summaries)
    development={s['method']:s for s in summaries if s['split']=='development'}
    seq,large,analytic=[development[k] for k in ['sequential_cp','fixed_cp_20000','analytic_cdf']]
    sample_efficiency_signal=(seq['false_safe']==0 and seq['unresolved']/seq['queries']<=large['unresolved']/large['queries']+.02
                              and seq['mean_batch_ms']<=.8*large['mean_batch_ms'])
    gate=dict(criteria=protocol['promotion_gate'],development_sample_efficiency_signal=bool(sample_efficiency_signal),
              analytic_exact_reference_deployable_on_this_family=True,
              analytic_mean_batch_ms=analytic['mean_batch_ms'],sequential_mean_batch_ms=seq['mean_batch_ms'],
              promote_adaptive_to_vehicle=False,
              reason='All tested predictions are independent isotropic Gaussians with a deployable exact CDF. No new adaptive-controller claim without benefit beyond that strong baseline and validation on non-analytic predictions.',
              no_navigation_holdout_claim=True)
    (a.output/'promotion_gate.json').write_text(json.dumps(gate,indent=2))
    bias=[]
    for f in fixtures:
        for j,truth in enumerate(f['reference']):
            for n in protocol['budgets']:
                vals=[r['estimate'] for r in rows if r['fixture']==f['id'] and r['query']==j and r['method']=='original_%d'%n]
                bias.append(dict(fixture=f['id'],split=f['split'],query=j,budget=n,
                                 structural_difference=float(f['density_reference'][j]-truth),
                                 sampling_mean_error_to_source_expression=float(np.mean(vals)-f['density_reference'][j]),
                                 sampling_std=float(np.std(vals,ddof=1))))
    save_csv(a.output/'bias_sampling_decomposition.csv',bias)
    result=dict(upstream_sha=UPSTREAM_SHA,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                fixture_banks=len(fixtures),unique_queries=sum(len(f['queries']) for f in fixtures),
                repeats=protocol['repeats'],rows=len(rows),summaries=summaries,gate=gate,
                scope='Constructed threshold-balanced independent isotropic Gaussian event queries. Holdout refers to frozen unseen constructed parameter combinations, not navigation maps or real-world risk.',
                confidence_scope='Simultaneous across8 fixed queries and4 predeclared looks per bank/repeat; conditional iid target-model draws. No global all-experiments or mission guarantee.',
                python=platform.python_version(),numpy=np.__version__,torch=torch.__version__)
    (a.output/'risk_budget_results.json').write_text(json.dumps(result,indent=2))
    print('RISK_BUDGET_DONE '+json.dumps(dict(banks=len(fixtures),rows=len(rows),gate=gate)),flush=True)


if __name__=='__main__':main()
