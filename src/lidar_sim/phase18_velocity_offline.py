"""Fixed-ego, same-detections synthetic velocity filter comparison.

The previous closed-loop experiment cannot attribute different *mean* errors
solely to a velocity estimator: diverged paths also change LiDAR geometry.
This offline diagnostic fixes the complete ego motion and checks the two
filters see bitwise identical noisy center measurements each timestamp.
Obstacle physics remains immutable; truth is used ONLY AFTER estimation to
score velocity and CV predictions. No vehicle commands are issued.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

from .evaluation import TrialSpec, make_trial
from .model import State
from .phase18_sim_eval import PROFILES
from .simulated_perception import SimulatedPerception

PAIRS = (("noisy", "noisy_ls5"), ("degraded", "degraded_ls5"))
HORIZONS = (0.5, 1.0, 2.0)


def stable_seed(spec: TrialSpec) -> int:
    return int.from_bytes(
        hashlib.sha256(spec.case_id.encode("utf-8")).digest()[:8], "big")


def fixed_ego_trajectory(*, dt: float, ticks: int,
                         cruise_speed: float = 1.5) -> list[State]:
    if (not math.isfinite(dt) or dt <= 0
            or not math.isfinite(cruise_speed) or cruise_speed < 0
            or isinstance(ticks, bool) or not isinstance(ticks, int)
            or ticks < 2):
        raise ValueError("invalid fixed-trajectory parameters")
    # Explicit counterfactual *straight* ego path; intentionally ignores the
    # planner and physical collisions, to isolate filter sensor noise.
    return [State(cruise_speed * (i * dt), 0., 0., cruise_speed,
                  time=i * dt) for i in range(ticks)]


def evaluate(specs: list[TrialSpec], *, dt: float, ticks: int
             ) -> tuple[list[dict], list[dict]]:
    if not specs:
        raise ValueError("at least one case required")
    ego = fixed_ego_trajectory(dt=dt, ticks=ticks)
    summary = []
    residuals = []
    for spec in specs:
        trial = make_trial(spec)
        reference = {x.name: x for x in trial.world.obstacles}
        seed = stable_seed(spec)
        for old_name, new_name in PAIRS:
            old_cfg, new_cfg = PROFILES[old_name], PROFILES[new_name]
            old_d = asdict(old_cfg)
            new_d = asdict(new_cfg)
            for field in ("velocity_estimator", "regression_window"):
                old_d.pop(field)
                new_d.pop(field)
            if old_d != new_d:
                raise AssertionError("sensor nuisance configurations disagree")
            a = SimulatedPerception(old_cfg, seed)
            b = SimulatedPerception(new_cfg, seed)
            total_valid = blocked = 0
            sums = {"ema": {"v": 0.0, "future": {h: 0.0 for h in HORIZONS}},
                    "ls5": {"v": 0.0, "future": {h: 0.0 for h in HORIZONS}}}
            for tick in range(ticks):
                history = ego[:tick+1]
                old = a.observe(trial.world, history, dt=dt)
                new = b.observe(trial.world, history, dt=dt)
                # Both estimators are observationally coupled on the same
                # fixed ego path. Never compare different sample sets.
                if (old.observation_count != new.observation_count
                        or old.blocked_reason != new.blocked_reason
                        or old.withheld_tracks != new.withheld_tracks
                        or old.stale_track_ids != new.stale_track_ids):
                    raise AssertionError("sensor measurements/availability diverged")
                old_targets = {o.name: o for o in old.estimated_world.obstacles}
                new_targets = {o.name: o for o in new.estimated_world.obstacles}
                if old_targets.keys() != new_targets.keys():
                    raise AssertionError("observed track IDs disagree")
                for name in old_targets:
                    x, y = old_targets[name], new_targets[name]
                    if (x.x0, x.y0, x.radius, x.reference_time_s) != (
                            y.x0, y.y0, y.radius, y.reference_time_s):
                        raise AssertionError("raw center observations differ")
                if old.blocked_reason:
                    blocked += 1
                    continue
                for name, old_est in old_targets.items():
                    new_est = new_targets[name]
                    truth = reference[name]
                    ve = {
                        "ema": math.hypot(old_est.vx-truth.vx,
                                          old_est.vy-truth.vy),
                        "ls5": math.hypot(new_est.vx-truth.vx,
                                          new_est.vy-truth.vy),
                    }
                    total_valid += 1
                    for method, value in ve.items():
                        sums[method]["v"] += value
                    for h in HORIZONS:
                        expected = truth.position(ego[tick].time + h)
                        errs = {}
                        for method, track in (("ema", old_est),
                                              ("ls5", new_est)):
                            px, py = track.position(ego[tick].time+h)
                            error = math.hypot(px-expected[0], py-expected[1])
                            sums[method]["future"][h] += error
                            errs[method] = error
                        residuals.append({
                            "case_id": spec.case_id,
                            "sensor_profile": old_name,
                            "observation_seed": seed,
                            "tick": tick,
                            "time_s": ego[tick].time,
                            "object_id": name,
                            "horizon_s": h,
                            "ema_velocity_error_mps": ve["ema"],
                            "ls5_velocity_error_mps": ve["ls5"],
                            "ema_cv_position_error_m": errs["ema"],
                            "ls5_cv_position_error_m": errs["ls5"],
                            "identical_noisy_centers": 1,
                            "fixed_ego_not_closed_loop": 1,
                        })
            row = {
                "case_id": spec.case_id,
                "scenario": spec.name,
                "stress": spec.stress,
                "seed": spec.seed,
                "reference_sensor": old_name,
                "candidate_sensor": new_name,
                "observation_seed": seed,
                "ticks": ticks,
                "blocked_ticks_both": blocked,
                "scored_track_ticks_both": total_valid,
                "ema_mean_velocity_error_mps": (
                    sums["ema"]["v"]/total_valid if total_valid else ""),
                "ls5_mean_velocity_error_mps": (
                    sums["ls5"]["v"]/total_valid if total_valid else ""),
            }
            for h in HORIZONS:
                for method in ("ema", "ls5"):
                    row[f"{method}_mean_cv_position_error_h{h:g}_m"] = (
                        sums[method]["future"][h]/total_valid
                        if total_valid else "")
            summary.append(row)
    return summary, residuals


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("empty fixed-trajectory output")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Offline, fixed-ego, same-LiDAR-noise estimator comparison")
    p.add_argument("--scenarios", default="two_targets")
    p.add_argument("--stress", default="weak_brake")
    p.add_argument("--seed-start", type=int, default=50)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--ticks", type=int, default=161)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--output", type=Path,
                   default=Path("results/velocity-fixed-ego"))
    args = p.parse_args()
    if args.seeds < 1 or args.seed_start < 0:
        p.error("seed range must be nonnegative and nonempty")
    names = args.scenarios.split(",")
    specs = [TrialSpec(name.strip(), seed, args.stress)
             for name in names for seed in range(
                 args.seed_start, args.seed_start+args.seeds)]
    rows, details = evaluate(specs, dt=args.dt, ticks=args.ticks)
    save_csv(args.output/"fixed_ego_pairs.csv", rows)
    if details:
        save_csv(args.output/"fixed_ego_residuals.csv", details)
    (args.output/"inputs.json").write_text(json.dumps({
        "specs": [asdict(s) for s in specs],
        "dt": args.dt, "ticks": args.ticks,
        "ego_path": "predeclared fixed straight line x=1.5*t,y=0,yaw=0",
        "profiles": [list(pair) for pair in PAIRS],
        "horizons_s": list(HORIZONS),
        "warning": (
            "This diagnostic isolates estimator noise with identical "
            "synthetic sensor measurements, not closed-loop robot safety."
        ),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    for row in rows:
        print(f"{row['case_id']} {row['reference_sensor']} -> "
              f"{row['candidate_sensor']} "
              f"n={row['scored_track_ticks_both']} "
              f"mean velocity error EMA={row['ema_mean_velocity_error_mps']} "
              f"LS5={row['ls5_mean_velocity_error_mps']}")
    print("Identical raw detections verified. This is NOT a controller rollout.")


if __name__ == "__main__":
    main()
