"""Simulation-only planning-envelope ablation with disjoint calibration seeds.

Calibration reads synthetic truth ONLY offline to estimate one unconditional,
episode-level max 0.5/1/2s CV center error. A finite split-conformal-style rank
is reported with all missing-observation exposure caveats. During closed-loop
control it injects only a scalar extra estimated-obstacle radius; the planner
never receives target truth, measured residuals or reference velocities.

This simple *static* inflation applies even to planning horizons >2s, so it is
NOT a calibrated long-horizon safety guarantee. Retain all negative outcomes.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path

from .evaluation import TrialSpec, make_trial, VALID_SCENARIOS, VALID_STRESSES
from .experiment import RunConfig, run, save_result
from .phase18_sim_eval import PROFILES
from .phase18_velocity_offline import evaluate as evaluate_fixed_ego, stable_seed
from .simulated_perception import SimulatedPerception


POLICIES = ("none", "fixed_030", "offline_q90_2s")
SENSOR_PROFILES = ("noisy_ls5", "degraded_ls5")
ALPHA = .10
FIXED_PADDING_M = .30


def rank_quantile(episode_errors: list[float], alpha: float) -> float:
    """Ceil((n+1)(1-alpha)) rank. Refuse a nonfinite/infinite upper bound."""
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly in (0,1)")
    if not episode_errors or any(
            not math.isfinite(v) or v < 0 for v in episode_errors):
        raise ValueError("nonempty finite nonnegative errors required")
    k = math.ceil((len(episode_errors)+1)*(1-alpha))
    if k > len(episode_errors):
        raise ValueError("insufficient independent calibration episodes")
    return sorted(episode_errors)[k-1]


def calibrate(specs: list[TrialSpec], *, ticks: int = 161,
              dt: float = .1, alpha: float = ALPHA) -> dict:
    if len({s.case_id for s in specs}) != len(specs):
        raise ValueError("calibration case IDs must be unique")
    if not specs:
        raise ValueError("calibration episodes required")
    _, residuals = evaluate_fixed_ego(specs, dt=dt, ticks=ticks)
    cases = {s.case_id for s in specs}
    result = {
        "alpha": alpha, "calibration_case_ids": sorted(cases),
        "calibration_ego_path": "fixed straight 1.5m/s, not a robot rollout",
        "scored_horizons_s": [0.5, 1.0, 2.0],
        "coverage_assumptions": (
            "conditional on scored valid tracks; no guarantees under "
            "occlusion/dropout, nonexchangeable episodes or horizons >2s"),
        "profiles": {},
    }
    for profile in SENSOR_PROFILES:
        grouped: dict[str, list[float]] = {case: [] for case in cases}
        count: dict[str, int] = {case: 0 for case in cases}
        for row in residuals:
            if row["sensor_profile"] != profile:
                continue
            case = row["case_id"]
            if case not in grouped:
                raise ValueError("leaked calibration case")
            grouped[case].append(float(row["ls5_cv_position_error_m"]))
            count[case] += 1
        if any(not grouped[case] for case in cases):
            raise ValueError(f"{profile}: calibration case missing scored tracks")
        maxima = {case: max(values) for case, values in grouped.items()}
        q = rank_quantile(list(maxima.values()), alpha)
        result["profiles"][profile] = {
            "extra_radius_m": q,
            "calibration_episode_units": len(cases),
            "scored_horizon_observations": sum(count.values()),
            "episode_max_residual_m": maxima,
            "method": "episode-max of all scored 0.5/1/2s CV center residuals",
            "not_formal_coverage": True,
        }
    return result


def run_study(specs: list[TrialSpec], *, calibration: dict,
              output: Path, max_time: float = 22.,
              modes: tuple[str, ...] = ("two_stage_viability",),
              profiles: tuple[str, ...] = SENSOR_PROFILES,
              policies: tuple[str, ...] = POLICIES) -> list[dict]:
    if not specs or not modes or not profiles or not policies:
        raise ValueError("nonempty evaluation dimensions required")
    if not math.isfinite(max_time) or max_time <= 0:
        raise ValueError("invalid max_time")
    if any(p not in SENSOR_PROFILES for p in profiles):
        raise ValueError("envelope study supports only LS5 sensor profiles")
    if any(m not in ("two_stage_viability", "two_stage_bodyframe")
           for m in modes):
        raise ValueError("unsupported planner mode")
    if any(p not in POLICIES for p in policies):
        raise ValueError("unsupported envelope policy")
    cal_ids = set(calibration["calibration_case_ids"])
    if len({s.case_id for s in specs}) != len(specs):
        raise ValueError("duplicate test cases")
    if any(s.case_id in cal_ids for s in specs):
        raise ValueError("calibration/test physical episode leakage")
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        trial = make_trial(spec)
        for profile in profiles:
            for policy in policies:
                padding = (
                    0.0 if policy == "none" else
                    FIXED_PADDING_M if policy == "fixed_030" else
                    float(calibration["profiles"][profile]["extra_radius_m"]))
                if not math.isfinite(padding) or padding < 0:
                    raise ValueError("invalid calibrated padding")
                for mode in modes:
                    observer = SimulatedPerception(
                        PROFILES[profile], stable_seed(spec),
                        obstacle_padding_m=padding)
                    out = run(spec.name, spec.seed, mode,
                              vehicle=trial.vehicle,
                              planner=trial.planner,
                              config=RunConfig(
                                  max_time=max_time,
                                  physical_road_footprint="rearaxle_rectangle",
                                  physical_body_offset_m=(
                                      trial.vehicle.rear_axle_to_body_center_m),
                                  goal_criterion="pose_gate",
                                  goal_lateral_tolerance_m=.75,
                                  goal_heading_tolerance_deg=15.),
                              world_override=trial.world,
                              perception=observer)
                    directory = output / spec.case_id / profile / policy / mode
                    save_result(out, directory)
                    rows.append({
                        "case_id": spec.case_id, "scenario": spec.name,
                        "stress": spec.stress, "seed": spec.seed,
                        "mode": mode, "sensor_profile": profile,
                        "policy": policy, "extra_obstacle_radius_m": padding,
                        "observation_seed": stable_seed(spec),
                        **out.metrics(),
                    })
    write_csv(output/"episodes.csv", rows)
    return rows


def pair_rows(rows: list[dict]) -> list[dict]:
    indexed: dict[tuple[str, str, str], dict[str, dict]] = {}
    for row in rows:
        key = (row["case_id"], row["mode"], row["sensor_profile"])
        group = indexed.setdefault(key, {})
        if row["policy"] in group:
            raise ValueError("duplicate policy within case")
        group[row["policy"]] = row
    paired = []
    for key, group in sorted(indexed.items()):
        if set(group) != set(POLICIES):
            raise ValueError(f"incomplete envelope policies for {key}")
        baseline = group["none"]
        for policy in POLICIES:
            current = group[policy]
            if current["observation_seed"] != baseline["observation_seed"]:
                raise ValueError("different observation seed in paired experiment")
            paired.append({
                "case_id": key[0], "mode": key[1],
                "sensor_profile": key[2], "policy": policy,
                "extra_obstacle_radius_m": current["extra_obstacle_radius_m"],
                "success": current["success"], "collision": current["collision"],
                "road_exit": current["road_exit"], "timeout": current["timeout"],
                "duration_s": current["duration_s"],
                "min_clearance_m": current["min_clearance_m"],
                "blocked_steps": current["perception_blocked_steps"],
                "planned_steps": current["perception_planned_steps"],
                "baseline_success": baseline["success"],
                "baseline_collision": baseline["collision"],
                "baseline_timeout": baseline["timeout"],
                "success_changed": current["success"] != baseline["success"],
                "collision_changed": current["collision"] != baseline["collision"],
            })
    return paired


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write empty experiment")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="2D simulation fixed-vs-offline-quantile occupancy padding")
    ap.add_argument("--scenario", default="two_targets",
                    choices=VALID_SCENARIOS)
    ap.add_argument("--stress", default="weak_brake",
                    choices=VALID_STRESSES)
    ap.add_argument("--calib-start", type=int, default=10)
    ap.add_argument("--calib-seeds", type=int, default=10)
    ap.add_argument("--test-start", type=int, default=60)
    ap.add_argument("--test-seeds", type=int, default=4)
    ap.add_argument("--ticks", type=int, default=161)
    ap.add_argument("--dt", type=float, default=.1)
    ap.add_argument("--max-time", type=float, default=22)
    ap.add_argument("--output", type=Path,
                    default=Path("results/phase18-envelope"))
    a = ap.parse_args()
    if (a.calib_start < 0 or a.test_start < 0 or
            a.calib_seeds < 1 or a.test_seeds < 1):
        ap.error("invalid independent seed splits")
    calibration_specs = [TrialSpec(a.scenario, i, a.stress) for i in
                         range(a.calib_start, a.calib_start+a.calib_seeds)]
    test_specs = [TrialSpec(a.scenario, i, a.stress) for i in
                  range(a.test_start, a.test_start+a.test_seeds)]
    if {s.case_id for s in calibration_specs} & {
            s.case_id for s in test_specs}:
        ap.error("calibration and test seed overlap")
    calibration = calibrate(calibration_specs, ticks=a.ticks, dt=a.dt)
    output = a.output
    output.mkdir(parents=True, exist_ok=True)
    (output/"calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2),
        encoding="utf-8")
    outcomes = run_study(test_specs, calibration=calibration,
                         output=output, max_time=a.max_time)
    paired = pair_rows(outcomes)
    write_csv(output/"paired.csv", paired)
    (output/"inputs.json").write_text(json.dumps({
        "calibration_spec": [asdict(s) for s in calibration_specs],
        "test_spec": [asdict(s) for s in test_specs],
        "max_time_s": a.max_time, "sensor_dt": a.dt,
        "fixed_ego_calibration_ticks": a.ticks,
        "policies": list(POLICIES),
        "fixed_radius_m": FIXED_PADDING_M,
        "warning": (
            "No long-horizon 90% collision probability guarantee: "
            "offline conditional 0.5/1/2s quantile is used as a constant "
            "padding for longer planner paths; physical outcomes are truth-only."
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in paired:
        result = ("success" if row["success"] else
                  "collision" if row["collision"] else
                  "road_exit" if row["road_exit"] else "timeout")
        print(f"{row['case_id']} {row['sensor_profile']}"
              f" {row['policy']} padding={row['extra_obstacle_radius_m']:.3f}m:"
              f" {result} t={row['duration_s']:.2f}s")
    print("Research diagnostic only, not formal safety coverage.")


if __name__ == "__main__":
    main()
