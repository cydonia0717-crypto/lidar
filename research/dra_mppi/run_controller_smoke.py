"""Unmodified upstream MPPI in a declared, custom headless test harness."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from run_upstream_audit import UPSTREAM_SHA, save_csv


def exact_unicycle(state, action, dt):
    # Independent plant: constant speed/yaw-rate arc, no Ackermann or actuator model.
    x, y, heading, _, progress, _ = state
    v, omega = map(float, action)
    end_heading = heading+omega*dt
    if abs(omega) < 1e-9:
        x += v*math.cos(heading)*dt
        y += v*math.sin(heading)*dt
    else:
        x += v/omega*(math.sin(end_heading)-math.sin(heading))
        y += v/omega*(-math.cos(end_heading)+math.cos(heading))
    return np.array([x, y, end_heading, v, progress+abs(v)*dt, omega])


def min_gap(state, action, means, velocities, now, dt, radius):
    # Dense independent event sampling with explicit between-node movement margin.
    times = np.linspace(0, dt, 101)
    ego = np.stack([exact_unicycle(state, action, float(t))[:2] for t in times])
    target = means[None]+(now+times[:, None, None])*velocities[None]
    distances = np.linalg.norm(ego[:, None, :]-target, axis=-1)
    # Relative motion is Lipschitz with bound |v_ego| + max |v_target|.
    margin = (abs(float(action[0]))+np.linalg.norm(velocities, axis=-1).max())*(dt/100)/2
    return float(distances.min()-radius), float(distances.min()-radius-margin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    upstream = args.upstream.resolve()
    actual = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == UPSTREAM_SHA
    source = upstream/'src/mppi_planner/src'
    sys.path.insert(0, str(source))
    from planner.mppi import MPPIPlanner
    from prob_mppi.dynamic_obstacles import DynamicObstacles
    from prob_mppi.jackal_dynamics import JackalDynamics
    from prob_mppi.jackal_objective import JackalObjective
    from utils.spline import TorchSpline2D
    base = yaml.safe_load((source/'prob_mppi/run_sim_jackal.yaml').read_text())
    base['device'] = base['mppi']['device'] = 'cpu'
    base['mppi']['horizon'] = 20
    base['mppi']['compute_cost_once'] = True
    base['mppi']['use_priors'] = False
    base['obstacles']['num_obstacles'] = 5  # Matches original hardcoded constructor.
    base['goal'] = [4., 0.]
    torch.set_num_threads(2)
    layouts = {
        'clear': (np.array([[20., 8.], [22., -8.], [24., 8.], [26., -8.], [28., 8.]]), np.zeros((5, 2))),
        'crossing': (np.array([[2., -2.], [22., -8.], [24., 8.], [26., -8.], [28., 8.]]), np.array([[0., 1.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]])),
        'head_on': (np.array([[4., 0.], [22., -8.], [24., 8.], [26., -8.], [28., 8.]]), np.array([[-.5, 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]])),
    }
    episodes, trace, commands = [], [], []

    def assemble(cfg, means):
        path = SimpleNamespace(x=[0., 2., 4., 6., 8.], y=[0., 0., 0., 0., 0.])
        spline = TorchSpline2D(path, device='cpu')
        n = 5
        obstacles = DynamicObstacles(cfg, torch.tensor(means[:, 0], dtype=torch.float32),
                                    torch.tensor(means[:, 1], dtype=torch.float32), torch.zeros(n),
                                    torch.eye(2).repeat(n, 1, 1)*.3, torch.zeros(n), torch.zeros(n))
        objective = JackalObjective(cfg, dyn_obstacles=obstacles, spline=spline)
        dynamics = JackalDynamics(cfg)
        planner = MPPIPlanner(cfg['mppi'], nx=6, dynamics=dynamics.step, running_cost=objective.compute)
        return planner, obstacles

    def predictions(obj, cfg, means, velocities, now):
        future = now+(np.arange(cfg['mppi']['horizon'])+1)*cfg['mppi']['dt']
        for j in range(5):
            points = means[j]+future[:, None]*velocities[j]
            # Explicit synthetic, unimodal standard deviation supplied to original squaring API.
            std = .3*np.sqrt(1+np.arange(cfg['mppi']['horizon'])*cfg['mppi']['dt'])
            factors = torch.eye(2)[None, None].repeat(cfg['mppi']['horizon'], 1, 1, 1)*torch.tensor(std, dtype=torch.float32)[:, None, None, None]
            obj.multi_mode_predicted_coordinates[j] = {
                'predicted_coordinates': torch.tensor(points[:, None], dtype=torch.float32),
                'predicted_covs': factors}

    # One full sample-count planning call; not a real-time performance claim.
    cfg = copy.deepcopy(base)
    cfg['mppi']['num_samples'] = 400
    cfg['obstacles']['N_monte_carlo'] = 20000
    np.random.seed(0)
    torch.manual_seed(0)
    planner, obstacles = assemble(cfg, layouts['crossing'][0])
    predictions(obstacles, cfg, *layouts['crossing'], 0.)
    started = time.perf_counter()
    action, samples, costs = planner.command(torch.zeros(6))
    full_call = {'K': 400, 'T': 20, 'N_mc': 20000, 'elapsed_cpu_s': time.perf_counter()-started,
                 'action_shape': list(action.shape), 'sample_shape': list(samples.shape),
                 'finite': bool(torch.isfinite(action).all() and torch.isfinite(costs).all())}
    assert full_call['finite']
    print('FULL_SIZE_CALL '+json.dumps(full_call), flush=True)
    del planner, obstacles

    for name, (means, velocities) in layouts.items():
        for seed in [0, 1]:
            cfg = copy.deepcopy(base)
            cfg['mppi']['num_samples'] = 64
            cfg['obstacles']['N_monte_carlo'] = 4096
            np.random.seed(seed)
            torch.manual_seed(seed)
            state = np.zeros(6)
            episode = {'scenario': name, 'seed': seed, 'outcome': 'timeout', 'K': 64, 'T': 20, 'N_mc': 4096,
                       'steps': 0, 'min_node_gap_m': None, 'min_certified_lower_gap_m': None,
                       'output_bound_violations': 0, 'event_resolution_warnings': 0}
            best_node, best_lower = float('inf'), float('inf')
            try:
                planner, obstacles = assemble(cfg, means)
                for step in range(60):
                    now = step*.2
                    predictions(obstacles, cfg, means, velocities, now)
                    started = time.perf_counter()
                    sequence, samples, costs = planner.command(torch.tensor(state, dtype=torch.float32))
                    elapsed = time.perf_counter()-started
                    if not torch.isfinite(sequence).all() or not torch.isfinite(costs).all():
                        raise FloatingPointError('Nonfinite controller output')
                    action = sequence[0].numpy().astype(float)
                    low, high = np.array(cfg['mppi']['u_min']), np.array(cfg['mppi']['u_max'])
                    # Save raw output: do not silently repair filtering/averaging with clipping.
                    bound_violation = bool(np.any(action < low-1e-6) or np.any(action > high+1e-6))
                    episode['output_bound_violations'] += bound_violation
                    node_gap, lower_gap = min_gap(state, action, means, velocities, now, .2, .825)
                    best_node, best_lower = min(best_node, node_gap), min(best_lower, lower_gap)
                    if lower_gap <= 0 < node_gap:
                        episode['event_resolution_warnings'] += 1
                    trace.append({'scenario': name, 'seed': seed, 'step': step, 'time_s': now,
                                  'x': state[0], 'y': state[1], 'heading': state[2],
                                  'command_v': action[0], 'command_omega': action[1],
                                  'node_gap_m': node_gap, 'interval_gap_lower_m': lower_gap,
                                  'cpu_planning_s': elapsed, 'bound_violation': bound_violation})
                    state = exact_unicycle(state, action, .2)
                    episode['steps'] = step+1
                    if node_gap <= 0:
                        episode['outcome'] = 'collision'
                        break
                    if lower_gap <= 0:
                        episode['outcome'] = 'event_unresolved'
                        break
                    if state[0] >= 4. and abs(state[1]) <= .5:
                        episode['outcome'] = 'success'
                        break
                episode['final_x'] = float(state[0])
                episode['final_y'] = float(state[1])
                episode['min_node_gap_m'] = best_node
                episode['min_certified_lower_gap_m'] = best_lower
            except Exception:
                episode['outcome'] = 'execution_error'
                episode['error'] = traceback.format_exc()
            episodes.append(episode)
            (args.output/'controller_episodes.json').write_text(json.dumps(episodes, indent=2), encoding='utf-8')
            print('EPISODE '+json.dumps(episode), flush=True)
    if trace:
        save_csv(args.output/'controller_trace.csv', trace)
    hashes = {}
    for relative in ['planner/mppi.py', 'planner/mppi_utils.py', 'prob_mppi/dynamic_obstacles.py',
                     'prob_mppi/jackal_dynamics.py', 'prob_mppi/jackal_objective.py', 'prob_mppi/cost_functions.py', 'utils/spline.py']:
        hashes[relative] = hashlib.sha256((source/relative).read_bytes()).hexdigest()
    result = {'upstream_sha': actual, 'source_sha256': hashes, 'full_size_planning_call': full_call,
              'episodes': episodes, 'outcomes': {k: sum(e['outcome']==k for e in episodes) for k in ['success', 'collision', 'timeout', 'execution_error', 'event_unresolved']},
              'scope': 'Custom 4m headless smoke scenarios using unmodified upstream MPPI, objective and dynamics; reduced closed-loop budget. Not ROS/Gazebo, social forces, original paper benchmark, real-car or Ackermann.',
              'success_rule': 'x>=4m and |y|<=0.5m; collision/event-unresolved precedence; deadline12s; no dwell task',
              'plant': 'Independent exact constant-speed/yaw-rate unicycle arc; noiseless fixed-velocity targets; disk collision radius0.825m; bounded-margin event check',
              'ros_navigation_reproduced': False, 'paper_tables_reproduced': False, 'new_algorithm_validated': False}
    (args.output/'controller_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('CONTROLLER_SUMMARY '+json.dumps(result['outcomes']), flush=True)
    if any(e['outcome']=='execution_error' for e in episodes):
        raise AssertionError('One or more controller episodes failed to execute; see raw errors')


if __name__ == '__main__':
    main()
