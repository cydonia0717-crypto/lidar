"""Frozen scripted-world mechanism diagnostics with unchanged upstream MPPI.

No ROS, learned predictor or Ackermann substitution. The only original-module
patch is the already-declared dynamic population count, in an output copy.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_upstream_audit import load_module, save_csv, UPSTREAM_SHA
from run_controller_smoke import exact_unicycle
from event_risk import exact_isotropic_union


class ScriptedWorld:
    """Evaluator-only truth. A predictor gets packets, never this object."""
    def __init__(self, scenario, seed):
        self.scenario = scenario
        self.shift = np.random.default_rng(seed).uniform(-.15, .15)
        self.branch = 1 if seed % 2 else -1

    def positions(self, times):
        t = np.asarray(times)
        base = np.array([[2.8+self.shift, -2.0], [25., 8.], [28., -8.], [30., 8.], [32., -8.]])
        out = np.broadcast_to(base, t.shape+(5,2)).copy()
        out[...,0,1] += .9*t
        after = np.maximum(t-1.4, 0)
        if self.scenario == 'stop': out[...,0,1] -= .9*after
        if self.scenario == 'accelerate': out[...,0,1] += .9*after
        if self.scenario == 'turn':
            out[...,0,1] -= .9*after
            out[...,0,0] -= .9*after
        if self.scenario == 'reverse': out[...,0,1] -= 1.8*after
        if self.scenario == 'mode_branch': out[...,0,0] += self.branch*.9*after
        if self.scenario == 'correlated':
            out[...,0,0] += .4*after
            out[...,1,:] = out[...,0,:]+np.array([.8,.4])
        return out


def packets(world, name, seed, dt, duration):
    times = np.arange(-5, int(duration/dt)+2)*dt
    noise = .20 if name == 'noise_under' else 0.
    rng = np.random.default_rng(seed+10000)
    errors = rng.normal(size=(len(times),5,2))*noise
    if name == 'heavy_tail': errors = np.clip(rng.standard_t(3,size=errors.shape)*.08,-.8,.8)
    values = world.positions(times)+errors
    keep = np.ones(len(times),bool)
    if name == 'dropout': keep &= ~((times>=1.2-1e-8)&(times<=2.6+1e-8))
    arrival = times+(.6 if name == 'delay' else 0.)
    return times[keep], arrival[keep], values[keep]


def predict(history, now, T, dt, ols=False):
    stamp, arrival, positions = history
    ids = np.flatnonzero(arrival <= now+1e-8)
    assert len(ids), 'No received packet: cannot silently read an undelivered observation'
    if len(ids)<2:  # Explicit stationary prior when velocity cannot be estimated.
        idx = ids[-1]
        return np.repeat(positions[idx][None],T,axis=0), float(now-stamp[idx]), True
    a,b = ids[-2:]
    velocity = (positions[b]-positions[a])/(stamp[b]-stamp[a])
    origin=positions[b]
    if ols and len(ids)>=3:
        recent=ids[-5:]
        x=stamp[recent]-stamp[recent].mean()
        velocity=(x[:,None,None]*positions[recent]).sum(0)/(x*x).sum()
        origin=positions[recent].mean(0)+(stamp[b]-stamp[recent].mean())*velocity
    future = now+(np.arange(T)+1)*dt
    return origin[None]+(future-stamp[b])[:,None,None]*velocity[None], float(now-stamp[b]), False


def rollout(state, controls, dt):
    states=[]
    for u in controls:
        state=exact_unicycle(state,np.clip(u,[0,-1],[2,1]),dt)
        states.append(state.copy())
    return np.array(states)


def model_risks(xy, means, sigma):
    # xy may be K,T,2 or T,2. No assumption of independence across time.
    from scipy.stats import ncx2
    d2=((xy[...,None,:]-means)**2).sum(-1)
    individual=ncx2.cdf(.825**2/sigma**2,2,d2/sigma**2)
    return 1-np.prod(1-individual,axis=-1)


def interval_event(world, state, u, now, dt):
    ts=np.linspace(0,dt,101)
    ego=np.array([exact_unicycle(state,u,float(s))[:2] for s in ts])
    targets=world.positions(now+ts)
    gaps=np.linalg.norm(ego[:,None]-targets,axis=-1)-.625
    wall=3.5-np.abs(ego[:,1])-.325
    node=float(min(gaps.min(),wall.min()))
    # All scripted target speeds <=1.8m/s, including branch/correlated motion.
    lower=node-(abs(float(u[0]))+1.8)*dt/200
    return node,lower,targets[-1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=Path(__file__).with_name('protocol.json'))
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(a.protocol.read_text()); (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    sha=subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()
    assert sha==UPSTREAM_SHA==protocol['upstream']
    src=a.upstream/'src/mppi_planner/src'; sys.path.insert(0,str(src))
    from planner.mppi import MPPIPlanner
    from prob_mppi.jackal_objective import JackalObjective
    from prob_mppi.jackal_dynamics import JackalDynamics
    from utils.spline import TorchSpline2D
    base=yaml.safe_load((src/'prob_mppi/run_sim_jackal.yaml').read_text())
    base['device']=base['mppi']['device']='cpu'; base['goal']=[8.,0.]
    base['mppi'].update(num_samples=64,horizon=20,compute_cost_once=True,use_priors=False)
    base['obstacles']['N_monte_carlo']=2048
    original=(src/'prob_mppi/dynamic_obstacles.py').read_text()
    assert original.count('self.n_obstacles = 5')==1
    copyfile=a.output/'dynamic_obstacles_count_compat.py'
    copyfile.write_text(original.replace('self.n_obstacles = 5','self.n_obstacles = len(x)',1))
    Obstacles=load_module(copyfile,'f_risk').DynamicObstacles
    torch.set_num_threads(2)

    class Objective(JackalObjective):
        def calculate_bound_dyn_cost(self, means, covs, t=None):
            if self.method=='dra': return super().calculate_bound_dyn_cost(means,covs,t)
            xy=torch.stack((self.x,self.y),-1).detach().numpy()
            mu=means.detach().numpy()
            if self.method=='event_cdf': risk=model_risks(xy,mu,self.sigma)
            else: risk=(np.linalg.norm(xy[...,None,:]-mu,axis=-1)<=.825).any(-1).astype(float)
            values=(risk>.05)*20000.+(100.*risk if self.method=='event_cdf' else 0.)
            return torch.tensor(values,dtype=self.x.dtype)
        def compute_running_cost(self,state,t=None,action=None,**kw):
            value=super().compute_running_cost(state,t,action,**kw)
            return value+(state[...,1].abs()>=3.175).float()*20000.

    def assemble(method,mu,sigma,seed,K=64,T=20,MC=2048):
        np.random.seed(seed); torch.manual_seed(seed)
        cfg=copy.deepcopy(base); cfg['mppi'].update(num_samples=K,horizon=T)
        cfg['obstacles'].update(num_obstacles=mu.shape[1],N_monte_carlo=MC)
        factor=torch.eye(2).repeat(mu.shape[1],1,1)*sigma
        obs=Obstacles(cfg,torch.tensor(mu[0,:,0]),torch.tensor(mu[0,:,1]),torch.zeros(mu.shape[1]),factor,torch.zeros(mu.shape[1]),torch.zeros(mu.shape[1]))
        spline=TorchSpline2D(SimpleNamespace(x=[0.,2.,4.,6.,8.],y=[0.]*5),device='cpu')
        objective=Objective(cfg,dyn_obstacles=obs,spline=spline);objective.method=method;objective.sigma=sigma
        dyn=JackalDynamics(cfg)
        planner=MPPIPlanner(cfg['mppi'],nx=6,dynamics=dyn.step,running_cost=objective.compute)
        return planner,obs,cfg

    def set_prediction(obs,mu,sigma):
        for j in range(mu.shape[1]):
            obs.multi_mode_predicted_coordinates[j]={'predicted_coordinates':torch.tensor(mu[:,j,None],dtype=torch.float32),
                'predicted_covs':torch.eye(2).repeat(len(mu),1,1,1)*sigma}

    def original_risk(mu,xy,sigma,seed,MC=2048):
        # Diagnostic RNG is independent and restored; cannot change planner trajectory.
        saved=torch.get_rng_state()
        try:
            torch.manual_seed(seed)
            cfg=copy.deepcopy(base);cfg['mppi'].update(num_samples=1,horizon=len(mu))
            cfg['obstacles']['N_monte_carlo']=MC
            n=mu.shape[1]
            ob=Obstacles(cfg,torch.tensor(mu[0,:,0]),torch.tensor(mu[0,:,1]),torch.zeros(n),torch.eye(2).repeat(n,1,1)*sigma,torch.zeros(n),torch.zeros(n))
            ob.create_multi_modal_gaussians(torch.tensor(mu[:,:,0],dtype=torch.float32),torch.tensor(mu[:,:,1],dtype=torch.float32),torch.eye(2).repeat(len(mu),n,1,1)*sigma)
            return ob.integrate_one_shot_monte_carlo_circles(torch.tensor(xy[None,:,0],dtype=torch.float32),torch.tensor(xy[None,:,1],dtype=torch.float32)).numpy()[0]
        finally: torch.set_rng_state(saved)

    # Basic independent world/event sanity checks; no navigation claim.
    w=ScriptedWorld('stop',101)
    assert np.allclose(w.positions([1.4])[0],w.positions([3.])[0])
    hist=packets(w,'stop',101,.2,16)
    mu,age,_=predict(hist,1.,20,.2)
    assert age<1e-8 and np.allclose(mu[0,0],w.positions([1.2])[0,0])
    assert interval_event(w,np.zeros(6),np.array([1.,0.]),0.,.2)[0] > 0
    episodes=[]; rows=[]
    detail=(a.output/'traces.jsonl').open('w')
    for name in protocol['scenarios']:
        for seed in protocol['seeds']:
            world=ScriptedWorld(name,seed)
            history=packets(world,name,seed,.2,protocol['deadline'])
            (a.output/f'packets_{name}_{seed}.json').write_text(json.dumps(dict(stamps=history[0].tolist(),arrivals=history[1].tolist(),positions=history[2].tolist())))
            sigma=.05 if name=='noise_under' else (.8 if name=='noise_over' else .3)
            variants=[(m,'cv') for m in protocol['methods']]
            if protocol.get('simple_controls'):
                variants += [(m,'ols5') for m in protocol['methods']]
                if name in ['noise_under','heavy_tail']:
                    variants += [(m,'cv_sigma08') for m in protocol['methods'] if m!='mean']
            if name in protocol['oracle_diagnostics']: variants.append(('event_cdf','oracle_future'))
            for method,predictor in variants:
                sigma=(.05 if name=='noise_under' else (.8 if name=='noise_over' else .3)) if predictor!='cv_sigma08' else .8
                identity=f'{name}_{seed}_{method}_{predictor}'
                state=np.zeros(6); times=[]; out='timeout'; mingap=math.inf; waiting=0
                mu,_,_=predict(history,0.,20,.2)
                planner,obs,cfg=assemble(method,mu,sigma,seed)
                for step in range(int(protocol['deadline']/.2)):
                    now=round(step*.2,8)
                    mu,age,prior=predict(history,now,20,.2,ols=predictor=='ols5')
                    if predictor=='oracle_future': mu=world.positions(now+(np.arange(20)+1)*.2)
                    set_prediction(obs,mu,sigma)
                    start=time.perf_counter();sequence,samples,costs=planner.command(torch.tensor(state,dtype=torch.float32));elapsed=time.perf_counter()-start
                    assert torch.isfinite(sequence).all() and torch.isfinite(costs).all()
                    controls=sequence.detach().numpy().astype(float)
                    u=np.clip(controls[0],[0,-1],[2,1]); future=rollout(state,controls,.2)
                    cdf=model_risks(future[:,:2],mu,sigma)
                    source=original_risk(mu,future[:,:2],sigma,900000+step)
                    sample_risk=model_risks(samples.detach().numpy(),mu[None],sigma)
                    sample_safe=int((sample_risk.max(axis=1)<=.05).sum())
                    true_future=world.positions(now+(np.arange(20)+1)*.2)
                    future_gap=np.linalg.norm(future[:,:2,None].transpose(0,2,1)-true_future,axis=-1)-.625
                    node,lower,truth_end=interval_event(world,state,u,now,.2)
                    mingap=min(mingap,node);waiting+=.2 if u[0]<.05 else 0.
                    row=dict(episode=identity,scenario=name,seed=seed,method=method,predictor=predictor,step=step,time=now,
                        age=age,startup_prior=prior,cpu_ms=elapsed*1000,source_risk_max=float(source.max()),cdf_risk_max=float(cdf.max()),
                        source_next=float(source[0]),cdf_next=float(cdf[0]),sample_safe_count=sample_safe,
                        output_risk_exceeds_threshold=bool(cdf.max()>.05),safe_samples_unsafe_output=bool(sample_safe and cdf.max()>.05),
                        interval_node_gap=node,interval_lower_gap=lower,next_node_contact=bool(np.linalg.norm(future[0,:2]-truth_end,axis=1).min()<=.625),
                        physical_contact=node<=0,future_openloop_node_gap=float(future_gap.min()),
                        model_truth_rmse=float(np.sqrt(np.mean((mu-true_future)**2))),clipped=bool(np.max(abs(u-controls[0]))>1e-6))
                    rows.append(row); times.append(elapsed*1000)
                    detail.write(json.dumps(dict(**row,state=state.tolist(),raw_controls=controls.tolist(),applied=u.tolist(),
                        observed_stamp=(now-age),predicted_means=mu.tolist(),sigma=sigma,oracle_truth_future=true_future.tolist(),
                        evaluated_output_xy=future[:,:2].tolist(),source_risk=source.tolist(),event_cdf=cdf.tolist(),truth_at_end=truth_end.tolist()))+'\n')
                    state=exact_unicycle(state,u,.2)
                    if node<=0: out='collision';break
                    if lower<=0: out='event_unresolved';break
                    if state[0]>=6 and abs(state[1])<=1:out='success';break
                e=dict(id=identity,scenario=name,seed=seed,method=method,predictor=predictor,outcome=out,steps=step+1,
                    time=(step+1)*.2,min_node_gap=mingap,waiting_s=waiting,mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),
                    truth_sha256=hashlib.sha256(world.positions(np.arange(0,16.01,.01)).tobytes()).hexdigest(),
                    observation_sha256=hashlib.sha256(b''.join(v.tobytes() for v in history)).hexdigest())
                episodes.append(e);print('F_EPISODE '+json.dumps(e),flush=True)
                (a.output/'episodes.json').write_text(json.dumps(episodes,indent=2))
    detail.close();save_csv(a.output/'control_metrics.csv',rows)
    # F3: matched-state, one-factor sweeps. Timings include planner command only.
    budgets=[(64,20,2048,8),(32,20,2048,8),(128,20,2048,8),(64,12,2048,8),(64,28,2048,8),
             (64,20,512,8),(64,20,8192,8),(64,20,2048,4),(64,20,2048,12)]
    if protocol.get('skip_budget_snapshots'): budgets=[]
    budgetrows=[]
    for K,T,MC,n in budgets:
        angles=np.arange(n)*2*np.pi/n
        centers=np.stack((2+.9*np.cos(angles),.9*np.sin(angles)),axis=-1)
        mu=np.repeat(centers[None],T,axis=0)
        for method in protocol['methods']:
            for seed in [11,22,33,44,55]:
                planner,obs,_=assemble(method,mu,.3,seed,K,T,MC);set_prediction(obs,mu,.3)
                start=time.perf_counter();seq,_,_=planner.command(torch.zeros(6));elapsed=time.perf_counter()-start
                xy=rollout(np.zeros(6),seq.numpy(),.2)[:,:2]
                cdf=model_risks(xy,mu,.3);source=original_risk(mu,xy,.3,seed,MC)
                budgetrows.append(dict(K=K,T=T,MC=MC,N=n,method=method,seed=seed,cpu_ms=1000*elapsed,
                    source_cdf_mae=float(abs(source-cdf).mean()),false_safe_steps=int(((source<=.05)&(cdf>.05)).sum()),
                    false_unsafe_steps=int(((source>.05)&(cdf<=.05)).sum()),cdf_risk_max=float(cdf.max()),
                    deadline5miss=elapsed>.005,deadline20miss=elapsed>.020,deadline50miss=elapsed>.050))
    if budgetrows: save_csv(a.output/'budget_snapshots.csv',budgetrows)
    summary=[]
    for name in protocol['scenarios']:
        for method in protocol['methods']:
            ee=[e for e in episodes if e['scenario']==name and e['method']==method and e['predictor']=='cv']
            summary.append(dict(scenario=name,method=method,**{k:sum(e['outcome']==k for e in ee) for k in ['success','collision','timeout','event_unresolved']}))
    save_csv(a.output/'summary.csv',summary)
    (a.output/'provenance.json').write_text(json.dumps(dict(upstream=sha,source_sha256={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob('*.py')},
        protocol_sha256=hashlib.sha256(a.protocol.read_bytes()).hexdigest(),episodes=len(episodes),controls=len(rows),budget_calls=len(budgetrows),
        scope='Development scripted-world diagnostics; oracle_future is diagnostic only. No new algorithm, no Ackermann, no closed-loop equal-time or unconditional calibration claim.'),indent=2))
    print('F_DONE '+json.dumps(dict(episodes=len(episodes),controls=len(rows),budget_calls=len(budgetrows))),flush=True)


if __name__=='__main__': main()
