"""Independent recorded-action replay and descriptive, episode-aware summaries."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def truth(name, anchor, branch, t):
    v=[[anchor,-2+.9*t],[25,8],[28,-8],[30,8],[32,-8]]
    h=max(0,t-1.4)
    if name=='stop': v[0][1]-=.9*h
    if name=='accelerate': v[0][1]+=.9*h
    if name=='turn': v[0][1]-=.9*h;v[0][0]-=.9*h
    if name=='reverse':v[0][1]-=1.8*h
    if name=='mode_branch':v[0][0]+=branch*.9*h
    if name=='correlated':v[0][0]+=.4*h;v[1]=[v[0][0]+.8,v[0][1]+.4]
    return v


def arc(s,u,t):
    x,y,h=s[:3];v,w=u;end=h+w*t
    if abs(w)<1e-9:return [x+v*math.cos(h)*t,y+v*math.sin(h)*t]
    return [x+v/w*(math.sin(end)-math.sin(h)),y-v/w*(math.cos(end)-math.cos(h))]


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    ee=json.loads((a.input/'episodes.json').read_text())
    per={e['id']:{'steps':0,'safe_samples_unsafe_output':0,'false_safe_source_steps':0,'risk_brier_sum':0.,'risk_events':0} for e in ee}
    anchors={};failures=[];checked=0
    # Initial truth anchor is recovered from the known t=.2 point in each trace;
    # switch only starts1.4s, so no RNG reimplementation is needed.
    with (a.input/'traces.jsonl').open() as f:
        for line in f:
            r=json.loads(line);key=r['episode'];t=r['time'];seed=r['seed'];name=r['scenario']
            if key not in anchors:anchors[key]=r['truth_at_end'][0][0]
            anchor=anchors[key];branch=1 if seed%2 else -1
            expected=truth(name,anchor,branch,t+.2)
            assert max(math.dist(x,y) for x,y in zip(expected,r['truth_at_end']))<1e-8
            xy=arc(r['state'],r['applied'],.2)
            assert math.dist(xy,r['evaluated_output_xy'][0])<1e-8
            node_event=min(math.dist(xy,o) for o in expected)<=.825
            d=per[key];d['steps']+=1
            d['safe_samples_unsafe_output']+=int(r['safe_samples_unsafe_output'])
            d['false_safe_source_steps']+=sum(x<=.05<y for x,y in zip(r['source_risk'],r['event_cdf']))
            d['risk_brier_sum']+=(r['cdf_next']-node_event)**2;d['risk_events']+=int(node_event)
            if r['physical_contact'] or r['interval_lower_gap']<=0:
                gaps=[]
                for j in range(101):
                    tau=j*.002;q=arc(r['state'],r['applied'],tau)
                    tt=truth(name,anchor,branch,t+tau)
                    gaps.append(min(min(math.dist(q,o)-.625 for o in tt),3.5-abs(q[1])-.325))
                assert abs(min(gaps)-r['interval_node_gap'])<1e-8
                failures.append(dict(episode=key,time=t,independent_min_gap=min(gaps),source_next=r['source_next'],cdf_next=r['cdf_next'],age=r['age'],
                    rmse=r['model_truth_rmse'],safe_samples=r['sample_safe_count'],
                    attribution='Evidence flags only; categories1/2/3/4/5 need paired counterfactuals; sampled evaluator replay passed(category6 not found at this terminal interval).'))
            checked+=1
    for e in ee:
        assert per[e['id']]['steps']==e['steps']
    for name in sorted({e['scenario'] for e in ee}):
        for seed in sorted({e['seed'] for e in ee}):
            same=[e for e in ee if e['scenario']==name and e['seed']==seed]
            assert len({e['truth_sha256'] for e in same})==1
            assert len({e['observation_sha256'] for e in same})==1
    summaries=[]
    for name in sorted({e['scenario'] for e in ee}):
        for method,predictor in sorted({(e['method'],e['predictor']) for e in ee if e['scenario']==name}):
            group=[e for e in ee if e['scenario']==name and e['method']==method and e['predictor']==predictor]
            rr=[per[e['id']] for e in group]
            summaries.append(dict(scenario=name,method=method,predictor=predictor,n=len(group),
                **{o:sum(e['outcome']==o for e in group) for o in ['success','collision','timeout','event_unresolved']},
                mean_episode_call_ms=statistics.mean(e['mean_ms'] for e in group),
                mean_wait_s=statistics.mean(e['waiting_s'] for e in group),
                safe_samples_unsafe_output=sum(v['safe_samples_unsafe_output'] for v in rr),
                descriptive_brier_next_planning_disk=sum(v['risk_brier_sum'] for v in rr)/sum(v['steps'] for v in rr)))
    def write_csv(path,rows):
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_csv(a.output/'episode_group_summary.csv',summaries)
    if failures:write_csv(a.output/'terminal_failure_replay.csv',failures)
    budgets=list(csv.DictReader((a.input/'budget_snapshots.csv').open())) if (a.input/'budget_snapshots.csv').exists() else []
    bs=[]
    for config in sorted({tuple(r[k] for k in ['K','T','MC','N','method']) for r in budgets}):
        group=[r for r in budgets if tuple(r[k] for k in ['K','T','MC','N','method'])==config]
        elapsed=sorted(float(r['cpu_ms']) for r in group)
        bs.append(dict(zip(['K','T','MC','N','method'],config),calls=len(group),mean_ms=statistics.mean(elapsed),
            p95_ms=elapsed[math.ceil(.95*len(elapsed))-1],
            false_safe_steps=sum(int(r['false_safe_steps']) for r in group),
            source_cdf_mae=statistics.mean(float(r['source_cdf_mae']) for r in group),
            **{f'miss_{ms}ms':sum(float(r['cpu_ms'])>ms for r in group) for ms in [5,20,50]}))
    if bs:write_csv(a.output/'budget_summary.csv',bs)
    result=dict(episodes=len(ee),controls_replayed=checked,terminal_intervals_replayed=len(failures),
        paired_truth_and_packets=True,summaries=summaries,
        boundaries=['All development;3 seeds per scenario, not broad statistical superiority.',
        'Brier pairs next-step model planning disk0.825 with same next-step disk event, not physical disk0.625 or mission collision.',
        'Scripted ensemble not sampled from provided Gaussian model: descriptive scores, no calibration guarantee.',
        'Source risk is independent posthoc reevaluation at same output queries, not original in-planner estimate.',
        'F3 snapshots have5 calls per configuration; P95 is small-sample upper order statistic, no tail latency certification.',
        'No equal-time closed-loop result, non-Gaussian/multimodal predictor or causal single-source failure proof yet.'],
        files_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.input.iterdir() if f.is_file() and f.suffix in ['.csv','.json','.jsonl']})
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['files_sha256','summaries']},indent=2))


if __name__=='__main__':main()
