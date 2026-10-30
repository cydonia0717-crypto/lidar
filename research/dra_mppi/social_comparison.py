"""Declared comparison adapters, not an author's unpublished baseline implementation."""
import torch


def make_objective_class(parent):
    class ComparisonObjective(parent):
        def __init__(self, *args, comparison=None, **kwargs):
            self.comparison = comparison
            self.last_risk = None
            super().__init__(*args, **kwargs)

        def calculate_bound_dyn_cost(self, obstacle_modes, obstacle_covs, t=None):
            if self.comparison['method'] == 'dra':
                return super().calculate_bound_dyn_cost(obstacle_modes, obstacle_covs, t)
            # The paper describes mean-only prediction, but does not release exact
            # baseline weights. This is our transparent deterministic adaptation.
            if self.x.ndim == 2:
                positions = torch.stack((self.x, self.y), dim=-1)[:, :, None, :]
                means = obstacle_modes[None, :, :, :]
            else:
                positions = torch.stack((self.x, self.y), dim=-1)[:, None, :]
                means = obstacle_modes[None, :, :]
            collision = (torch.linalg.norm(positions-means, dim=-1) <= .825).any(dim=-1)
            return collision.to(self.x.dtype)*20000.

        def compute_running_cost(self, state, t=None, action=None, **kwargs):
            cost = super().compute_running_cost(state, t, action, **kwargs)
            if self.comparison['wall_cost']:
                # Shared radius-aware scenario boundary, finite penalty only.
                cost = cost + (torch.abs(state[..., 1]) >= 3.5-.325).to(cost.dtype)*20000.
            return cost
    return ComparisonObjective


class ExactBatchUnicycle:
    """Prediction-only diagnostic; matches the separately implemented ego arc."""
    def __init__(self, cfg):
        self.dt = cfg['mppi']['dt']

    def step(self, states, actions, t=None):
        assert actions.ndim == 3
        v, omega = actions[..., 0], actions[..., 1]
        dtheta = omega*self.dt
        heading_end = states[:, None, 2] + torch.cumsum(dtheta, dim=1)
        heading_mid = heading_end-dtheta/2
        # torch.sinc(z)=sin(pi*z)/(pi*z), with a stable zero-yaw limit.
        length = v*self.dt*torch.sinc(dtheta/(2*torch.pi))
        x = states[:, None, 0]+torch.cumsum(length*torch.cos(heading_mid), dim=1)
        y = states[:, None, 1]+torch.cumsum(length*torch.sin(heading_mid), dim=1)
        progress = states[:, None, 4]+torch.cumsum(torch.abs(v)*self.dt, dim=1)
        return torch.stack((x, y, heading_end, v, progress, omega), dim=-1), actions


def output_wall_audit(state, sequence, unfiltered, sampled_xy, dt, exact_step):
    # No new planner or risk calls; does not consume any RNG state.
    def wall(sequence_):
        current = state.copy()
        minimum = 3.5-abs(float(current[1]))-.325
        for action in sequence_:
            action = action.clip([0., -1.], [2., 1.])
            for _ in range(4):
                current = exact_step(current, action, dt/4)
                minimum = min(minimum, 3.5-abs(float(current[1]))-.325)
        return float(minimum)
    return {
        'filtered_exact_plan_wall_gap_m': wall(sequence),
        'unfiltered_exact_plan_wall_gap_m': wall(unfiltered),
        'sampled_plans_wall_violation_count': int((sampled_xy[..., 1].abs() >= 3.175).any(dim=1).sum()),
        'filter_max_change': float(abs(sequence-unfiltered).max()),
    }


def self_check(parent, cfg, spline, exact_step):
    import numpy as np
    from types import SimpleNamespace
    cls = make_objective_class(parent)
    probe = dict(cfg)
    probe['mppi'] = dict(cfg['mppi'], num_samples=2)
    objective = cls(probe, spline=spline, comparison={'method': 'mean', 'wall_cost': True})
    objective.x = torch.tensor([[0., 2.], [0., 2.]])
    objective.y = torch.zeros(2, 2)
    means = torch.tensor([[[0., 0.]], [[4., 0.]]])
    got = objective.calculate_bound_dyn_cost(means, None)
    assert torch.equal(got, torch.tensor([[20000., 0.], [20000., 0.]]))
    state = torch.zeros(2, 2, 6)
    state[1, :, 1] = 3.2
    control = torch.zeros(2, 2, 2)
    objective.dyn_obstacles = None
    objective.comparison['wall_cost'] = False
    without = objective.compute_running_cost(state, action=control)
    objective.comparison['wall_cost'] = True
    with_wall = objective.compute_running_cost(state, action=control)
    assert torch.allclose(with_wall-without, torch.tensor([[0., 0.], [20000., 20000.]]))
    initial = np.array([1., .2, .4, 0., 0., 0.])
    actions = np.array([[1.2, 0.], [1.8, .7], [.5, -1.], [0., .2]])
    exact, _ = ExactBatchUnicycle(cfg).step(torch.tensor(initial[None], dtype=torch.float64),
                                          torch.tensor(actions[None], dtype=torch.float64))
    expected, current = [], initial
    for action in actions:
        current = exact_step(current, action, cfg['mppi']['dt'])
        expected.append(current)
    assert np.allclose(exact.numpy()[0], expected, atol=1e-12)
    return {'mean_cost_and_wall_delta': 'passed', 'exact_batch_vs_independent_arc': 'passed'}
