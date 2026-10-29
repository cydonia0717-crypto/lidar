"""Execute pinned upstream modules; this is not a navigation reproduction."""
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.stats import ncx2
import torch
import yaml

UPSTREAM_SHA = "6d76ab2878c503121544b992c0e5706c6c6dece0"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_obstacles(cls, means, sigma, k, mc_samples):
    """Inputs to upstream's API: sigma*I; upstream squares it elementwise."""
    horizon, n, _ = means.shape
    cfg = {
        "device": "cpu", "start": [0.0, 0.0, 0.0],
        "mppi": {"device": "cpu", "num_samples": k, "horizon": horizon},
        "obstacles": {"print_time": False, "use_gaussian_batch": True,
                      "num_obstacles": n, "N_monte_carlo": mc_samples,
                      "integral_radius": 0.3, "sample_bound": 5.0},
    }
    cov_input = torch.eye(2, dtype=torch.float32).repeat(horizon, n, 1, 1) * sigma
    obj = cls(cfg, means[0, :, 0], means[0, :, 1], torch.zeros(n),
              cov_input[0], torch.zeros(n), torch.zeros(n))
    obj.create_multi_modal_gaussians(means[:, :, 0], means[:, :, 1], cov_input)
    return obj, cfg


def exact_independent_event_probability(means, queries, sigma, radius):
    # Each isotropic Gaussian disk probability has a noncentral chi-square CDF.
    # This oracle first integrates each obstacle, then forms the union of events.
    delta = queries[:, :, None, :] - means[None, :, :, :]
    noncentrality = np.sum(delta ** 2, axis=-1) / sigma ** 2
    individual = ncx2.cdf(radius ** 2 / sigma ** 2, df=2, nc=noncentrality)
    return 1.0 - np.prod(1.0 - individual, axis=-1)


def audit_dynamics(cls, out):
    cfg = {"device": "cpu", "mppi": {"dt": 0.2}}
    model = cls(cfg)
    state = torch.zeros((1, 6), dtype=torch.float32)
    straight = torch.tensor([[[1.0, 0.0]] * 20])
    batch, _ = model.step(state, straight)
    expected_x = torch.arange(1, 21) * 0.2
    error = float(torch.max(torch.abs(batch[0, :, 0] - expected_x)))
    if error > 1e-5:
        raise AssertionError(f"Straight motion reference mismatch: {error}")
    turning = torch.tensor([[[1.0, 0.5]] * 20])
    batch, _ = model.step(state, turning)
    sequential = []
    current = state.clone()
    for t in range(turning.shape[1]):
        current, _ = model.step(current, turning[:, t, :])
        sequential.append(current[0].clone())
    sequential = torch.stack(sequential)
    rows = []
    for t in range(20):
        rows.append({"step": t + 1, "time_s": (t + 1) * 0.2,
                     "batch_x": float(batch[0, t, 0]), "batch_y": float(batch[0, t, 1]),
                     "sequential_x": float(sequential[t, 0]),
                     "sequential_y": float(sequential[t, 1])})
    save_csv(out / "dynamics_paths.csv", rows)
    return {"straight_max_abs_error_m": error,
            "turning_batch_vs_sequential_max_position_difference_m": float(
                torch.linalg.vector_norm(batch[0, :, :2] - sequential[:, :2], dim=1).max()),
            "interpretation": "Two original API paths, no patch. Difference is a reproduction audit, not an innovation or collision result."}


def audit_risk(cls, out):
    rows, summaries = [], []
    seeds = [0, 1, 2]
    for seed in seeds:
        torch.manual_seed(seed)
        means = torch.zeros((1, 1, 2), dtype=torch.float32)
        x = torch.tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
        y = torch.zeros_like(x)
        obj, cfg = make_obstacles(cls, means, 0.5, 5, 20000)
        estimate = obj.integrate_one_shot_monte_carlo_circles(x, y).numpy()
        queries = torch.stack((x, y), dim=-1).numpy()
        reference = exact_independent_event_probability(means.numpy(), queries, 0.5, 0.3)
        max_error = float(np.abs(estimate - reference).max())
        if not np.isfinite(estimate).all() or max_error > 0.02:
            raise AssertionError(f"Single Gaussian execution/reference check failed: seed={seed}, error={max_error}")
        summaries.append({"fixture": "single_isotropic_gaussian", "seed": seed,
                          "max_abs_error": max_error, "execution_check_passed": True})
        for k in range(5):
            rows.append({"fixture": "single_isotropic_gaussian", "seed": seed,
                         "obstacles": 1, "sigma_m": 0.5, "candidate": k, "step": 0,
                         "upstream_value": float(estimate[k, 0]),
                         "independent_event_reference": float(reference[k, 0])})
    # Constructed module fixtures, not random navigation episodes or a held-out benchmark.
    for n in [4, 8, 12]:
        for sigma in [0.6, 0.2]:
            for seed in seeds:
                torch.manual_seed(seed)
                horizon, k = 20, 16
                angle = torch.arange(n, dtype=torch.float32) * (2.0 * math.pi / n)
                base = torch.stack((0.55 * torch.cos(angle), 0.55 * torch.sin(angle)), dim=-1)
                means = base.unsqueeze(0).repeat(horizon, 1, 1)
                x = torch.linspace(-0.8, 0.8, k).unsqueeze(1).repeat(1, horizon)
                y = torch.zeros_like(x)
                obj, cfg = make_obstacles(cls, means, sigma, k, 20000)
                started = time.perf_counter()
                estimate = obj.integrate_one_shot_monte_carlo_circles(x, y).numpy()
                elapsed = time.perf_counter() - started
                queries = torch.stack((x, y), dim=-1).numpy()
                reference = exact_independent_event_probability(means.numpy(), queries, sigma, 0.3)
                if not np.isfinite(estimate).all():
                    raise AssertionError("Nonfinite upstream output")
                summaries.append({"fixture": "independent_gaussian_ring", "seed": seed,
                                  "obstacles": n, "sigma_m": sigma, "shape": list(estimate.shape),
                                  "upstream_min": float(estimate.min()), "upstream_max": float(estimate.max()),
                                  "outside_probability_range_count": int(((estimate < 0) | (estimate > 1)).sum()),
                                  "max_abs_difference_from_event_union": float(np.abs(estimate - reference).max()),
                                  "cpu_kernel_elapsed_s": elapsed,
                                  "note": "Finite output is an execution check; agreement with probability semantics is reported, not assumed."})
                for candidate in range(k):
                    for t in range(horizon):
                        rows.append({"fixture": "independent_gaussian_ring", "seed": seed,
                                     "obstacles": n, "sigma_m": sigma, "candidate": candidate, "step": t,
                                     "upstream_value": float(estimate[candidate, t]),
                                     "independent_event_reference": float(reference[candidate, t])})
    save_csv(out / "risk_values.csv", rows)
    return summaries


def run(upstream, out):
    sha = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if sha != UPSTREAM_SHA:
        raise AssertionError(f"Unexpected upstream commit {sha}")
    core = upstream / "src/mppi_planner/src/prob_mppi"
    paths = [core / "dynamic_obstacles.py", core / "jackal_dynamics.py",
             core / "run_sim_jackal.yaml", upstream / ".devcontainer/Dockerfile",
             upstream / "src/mppi_planner/src/ros_jackalsimulator.py"]
    hashes = {str(p.relative_to(upstream)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for p in paths:
        if p.suffix == ".py":
            ast.parse(p.read_text())
    cfg = yaml.safe_load((core / "run_sim_jackal.yaml").read_text())
    torch.set_num_threads(2)
    np.random.seed(0)
    torch.manual_seed(0)
    dynamics = load_module(core / "jackal_dynamics.py", "upstream_dynamics").JackalDynamics
    obstacles = load_module(core / "dynamic_obstacles.py", "upstream_obstacles").DynamicObstacles
    result = {
        "scope": "Unmodified upstream Python dynamics and risk modules on CPU, constructed fixtures only",
        "upstream_commit": sha, "source_sha256": hashes,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "numpy": np.__version__,
                        "cuda_available": torch.cuda.is_available(), "cpu_threads": torch.get_num_threads()},
        "status": {"module_execution_complete": False, "full_navigation_reproduction": False,
                   "paper_table_reproduction": False, "ackermann_adaptation": False,
                   "new_algorithm_validated": False, "navigation_episodes": 0},
        "config_audit": {"active_config": "src/mppi_planner/src/prob_mppi/run_sim_jackal.yaml",
                         "repository_horizon": cfg["mppi"]["horizon"], "paper_horizon": 20,
                         "repository_num_samples": cfg["mppi"]["num_samples"],
                         "repository_num_obstacles": cfg["obstacles"]["num_obstacles"],
                         "risk_api_covariance_convention": "Source uses elementwise square. Fixtures pass diagonal standard deviations explicitly.",
                         "constructor_obstacle_count": "Original constructor hardcodes 5; batch fixtures exercise direct Gaussian API, not ROS callback or multi-mode bookkeeping."},
        "dynamics": audit_dynamics(dynamics, out), "risk": audit_risk(obstacles, out),
    }
    result["status"]["module_execution_complete"] = True
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# DRA-MPPI upstream module execution", "",
             "Original upstream modules executed on CPU. Navigation episodes: **0**.",
             "Full ROS navigation, paper tables, Ackermann adaptation and GPU real-time results: **not completed**.", "",
             "The multi-obstacle reference integrates each independent obstacle event first, then forms its union. Differences from the upstream output are preserved in risk_values.csv; a green job is not a safety/correctness certificate.", "",
             f"Upstream: {sha}",
             f"Single-Gaussian checks: {sum(x['fixture'] == 'single_isotropic_gaussian' for x in result['risk'])} seeded fixtures.",
             f"Multi-obstacle diagnostics: {sum(x['fixture'] == 'independent_gaussian_ring' for x in result['risk'])} seeded fixtures.",
             f"Turning API path maximum position difference: {result['dynamics']['turning_batch_vs_sequential_max_position_difference_m']:.8f} m.", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"module_execution_complete": True, "navigation_episodes": 0,
                      "risk_fixtures": len(result["risk"]), "upstream_commit": sha}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        run(args.upstream.resolve(), args.output.resolve())
    except Exception:
        (args.output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
