"""Phase18 time-indexed experimental occupancy margins in 2D simulation.

The margin is a linear function of prediction time from CURRENT decision, not
observation age. We compare:
  none: no extra radius
  fixed_030: +0.30 m at every future time
  time_ramp: predefined engineering heuristic +0.10..+0.50m
  time_q90: per-horizon episode-level, offline, independent-seed quantiles

This is a geometry ablation, not a formal prediction set: per-horizon ranks
do not guarantee joint coverage; missing targets are not covered; post-2s
planner brake/hold paths inherit the last node with no calibrated guarantee.
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
from .phase18_envelope_eval import rank_quantile, FIXED_PADDING_M
from .phase18_sim_eval import PROFILES
from .phase18_velocity_offline import evaluate as fixed_ego_errors, stable_seed
from .simulated_perception import SimulatedPerception


HORIZONS = (0.0, 0.5, 1.0, 2.0)
RAMP_M = (0.10, 0.20, 0.30, 0.50)
POLICIES = ("none", "fixed_030", "time_ramp", "time_q90")
SENSOR_PROFILES = ("noisy_ls5", "degraded_ls5")
ALPHA = .10


def calibrate_horizons(specs: list[TrialSpec], *,
                       dt: float = .1, ticks: int = 161,
                       alpha: float = ALPHA) -> dict:
    if not specs or len({s.case_id for s in specs}) != len(specs):
        raise ValueError("nonempty unique calibration episodes required")
    _, records = fixed_ego_errors(specs, dt=dt, ticks=ticks)
    case_ids = {s.case_id for s in specs}
    result = {
        "calibration_case_ids": sorted(case_ids),
        "calibration_dt_s": dt,
        "calibration_ticks": ticks,
        "score_horizons_s": list(HORIZONS[1:]),
        "alpha_per_horizon": alpha,
        "warning": (
            "Per-horizon conditional scored-track quantiles do not certify "
            "joint 0.5/1/2s coverage, unobserved obstacles, or long-horizon "
            "braking/hold collision freedom. All reference truth is offline."
        ),
        "profiles": {},
    }
    for profile in SENSOR_PROFILES:
        source = profile.removesuffix("_ls5")
        episode_max: dict[float, dict[str, float]] = {
            h: {} for h in HORIZONS[1:]}
        counts: dict[float, int] = {h: 0 for h in HORIZONS[1:]}
        for r in records:
            if r["sensor_profile"] != source:
                continue
            case = r["case_id"]
            if case not in case_ids:
                raise ValueError("reference case outside calibration split")
            h = float(r["horizon_s"])
            if h not in episode_max:
                raise ValueError("unexpected horizon in offline score")
            v = float(r["ls5_cv_position_error_m"])
            if not math.isfinite(v) or v < 0:
                raise ValueError("bad calibrated position residual")
            episode_max[h][case] = max(episode_max[h].get(case, 0.), v)
            counts[h] += 1
        q = {}
        for h in HORIZONS[1:]:
            if set(episode_max[h]) != case_ids:
                raise ValueError(f"{profile} horizon={h} missing scored episodes")
            q[h] = rank_quantile(list(episode_max[h].values()), alpha)
        # Larger h must never get a smaller extra radius in this experiment:
        # this is a geometric monotonicity choice, not a coverage theorem.
        q05, q10, q20 = q[.5], max(q[.5], q[1.0]), max(q[.5], q[1.0], q[2.0])
        schedule = (q05, q05, q10, q20)
        result["profiles"][profile] = {
            "source_offline_profile": source,
            "time_nodes_s": list(HORIZONS),
            "padding_m": list(schedule),
            "raw_per_horizon_q_m": {str(h): q[h] for h in HORIZONS[1:]},
            "episode_max_error_m": {str(h): episode_max[h] for h in HORIZONS[1:]},
            "scored_track_horizon_events": {str(h): counts[h] for h in HORIZONS[1:]},
            "calibration_episode_units": len(case_ids),
            "method": "independent-case max-scored-residual per horizon, alpha=.1",
            "joint_coverage_certified": False,
        }
    return result


def study(specs: list[TrialSpec], *, calibration: dict,
          output: Path, max_time: float = 22.,
          profiles: tuple[str, ...] = SENSOR_PROFILES,
          policies: tuple[str, ...] = POLICIES,
          modes: tuple[str, ...] = ("two_stage_viability",)) -> list[dict]:
    if not specs or not profiles or not policies or not modes:
        raise ValueError("empty experimental dimensions")
    if (not math.isfinite(max_time) or max_time <= 0
            or any(p not in SENSOR_PROFILES for p in profiles)
            or any(p not in POLICIES for p in policies)
            or any(m not in ("two_stage_viability", "two_stage_bodyframe")
                   for m in modes)):
        raise ValueError("invalid experimental dimensions")
    case_ids = [s.case_id for s in specs]
    if (len(set(case_ids)) != len(case_ids) or
            set(case_ids) & set(calibration["calibration_case_ids"])):
        raise ValueError("duplicate or leaked calibration/test cases")
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        trial = make_trial(spec)
        for profile in profiles:
            for policy in policies:
                static_padding = FIXED_PADDING_M if policy == "fixed_030" else 0.
                horizon_nodes = ()
                schedule = ()
                if policy == "time_ramp":
                    horizon_nodes = HORIZONS
                    schedule = RAMP_M
                elif policy == "time_q90":
                    cal = calibration["profiles"][profile]
                    horizon_nodes = tuple(cal["time_nodes_s"])
                    schedule = tuple(cal["padding_m"])
                for mode in modes:
                    observer = SimulatedPerception(
                        PROFILES[profile], stable_seed(spec),
                        obstacle_padding_m=static_padding,
                        time_padding_horizons_s=horizon_nodes,
                        time_padding_values_m=schedule)
                    result = run(
                        spec.name, spec.seed, mode, vehicle=trial.vehicle,
                        planner=trial.planner,
                        config=RunConfig(
                            max_time=max_time,
                            physical_road_footprint="rearaxle_rectangle",
                            physical_body_offset_m=(
                                trial.vehicle.rear_axle_to_body_center_m),
                            goal_criterion="pose_gate",
                            goal_lateral_tolerance_m=.75,
                            goal_heading_tolerance_deg=15.),
                        world_override=trial.world, perception=observer)
                    save_result(result, output / spec.case_id / profile / policy / mode)
                    rows.append({
                        "case_id": spec.case_id,
                        "scenario": spec.name,
                        "stress": spec.stress,
                        "seed": spec.seed,
                        "mode": mode,
                        "sensor_profile": profile,
                        "policy": policy,
                        "observation_seed": stable_seed(spec),
                        "extra_radius_at_h0_m": (
                            schedule[0] if schedule else static_padding),
                        "extra_radius_at_h05_m": (
                            schedule[1] if schedule else static_padding),
                        "extra_radius_at_h1_m": (
                            schedule[2] if schedule else static_padding),
                        "extra_radius_at_h2_m": (
                            schedule[3] if schedule else static_padding),
                        "extra_radius_beyond_2s_m": (
                            schedule[-1] if schedule else static_padding),
                        **result.metrics(),
                    })
    write_csv(output/"episodes.csv", rows)
    return rows


def paired(rows: list[dict], *, policies: tuple[str, ...] = POLICIES
           ) -> list[dict]:
    groups: dict[tuple[str, str, str], dict[str, dict]] = {}
    for row in rows:
        key = (row["case_id"], row["mode"], row["sensor_profile"])
        slot = groups.setdefault(key, {})
        if row["policy"] in slot:
            raise ValueError("duplicate same-policy paired episode")
        slot[row["policy"]] = row
    output = []
    for key, slot in sorted(groups.items()):
        if set(slot) != set(policies) or "none" not in slot:
            raise ValueError(f"incomplete paired policy set: {key}")
        baseline = slot["none"]
        for policy in policies:
            r = slot[policy]
            if r["observation_seed"] != baseline["observation_seed"]:
                raise ValueError("paired case uses different observation seeds")
            output.append({
                "case_id": key[0],
                "mode": key[1], "sensor_profile": key[2],
                "policy": policy, "success": r["success"],
                "collision": r["collision"], "timeout": r["timeout"],
                "road_exit": r["road_exit"],
                "duration_s": r["duration_s"],
                "min_clearance_m": r["min_clearance_m"],
                "stopped_steps": r["stopped_steps"],
                "perception_blocked_steps": r["perception_blocked_steps"],
                "extra_radius_at_h0_m": r["extra_radius_at_h0_m"],
                "extra_radius_at_h1_m": r["extra_radius_at_h1_m"],
                "extra_radius_at_h2_m": r["extra_radius_at_h2_m"],
                "baseline_success": baseline["success"],
                "baseline_collision": baseline["collision"],
                "success_changed": r["success"] != baseline["success"],
                "collision_changed": r["collision"] != baseline["collision"],
            })
    return output


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError("cannot save empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def main() -> None:
    p = argparse.ArgumentParser(description="Phase18 2D time-dependent occupancy-margin ablation")
    p.add_argument("--scenario", choices=VALID_SCENARIOS, default="two_targets")
    p.add_argument("--stress", choices=VALID_STRESSES, default="weak_brake")
    p.add_argument("--calib-start", type=int, default=10)
    p.add_argument("--calib-seeds", type=int, default=10)
    p.add_argument("--test-start", type=int, default=60)
    p.add_argument("--test-seeds", type=int, default=4)
    p.add_argument("--dt", type=float, default=.1)
    p.add_argument("--ticks", type=int, default=161)
    p.add_argument("--max-time", type=float, default=22.)
    p.add_argument("--output", type=Path,
                   default=Path("results/phase18-time-envelope"))
    a = p.parse_args()
    if min(a.calib_start, a.test_start) < 0 or min(
            a.calib_seeds, a.test_seeds) < 1:
        p.error("bad calibration/test seeds")
    cal = [TrialSpec(a.scenario, seed, a.stress) for seed in range(
        a.calib_start, a.calib_start+a.calib_seeds)]
    tests = [TrialSpec(a.scenario, seed, a.stress) for seed in range(
        a.test_start, a.test_start+a.test_seeds)]
    config = calibrate_horizons(cal, ticks=a.ticks, dt=a.dt)
    output = a.output
    output.mkdir(parents=True, exist_ok=True)
    (output/"calibration_by_horizon.json").write_text(json.dumps(
        config, ensure_ascii=False, indent=2), encoding="utf-8")
    results = study(tests, calibration=config, output=output,
                    max_time=a.max_time)
    write_csv(output/"paired.csv", paired(results))
    (output/"inputs.json").write_text(json.dumps({
        "calibration_specs": [asdict(s) for s in cal],
        "test_specs": [asdict(s) for s in tests],
        "calibration_ticks": a.ticks, "dt": a.dt, "max_time": a.max_time,
        "policies": list(POLICIES),
        "heuristic_time_nodes_s": list(HORIZONS),
        "heuristic_ramp_m": list(RAMP_M),
        "warning": (
            "Per-horizon quantiles lack a joint guarantee; after 2s last "
            "radius is held constant for longer braking/hold trajectories. "
            "Missing tracks and physical safety are not calibrated."
        ),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in results:
        state = ("success" if r["success"] else "collision"
                 if r["collision"] else "road_exit"
                 if r["road_exit"] else "timeout")
        print(f"{r['case_id']} {r['sensor_profile']} {r['policy']}: "
              f"{state}, h0={r['extra_radius_at_h0_m']:.3f} "
              f"h2={r['extra_radius_at_h2_m']:.3f} "
              f"t={r['duration_s']:.2f}")
    print(f"Saved {len(results)} truth-scored synthetic episodes, no safety guarantee.")


if __name__ == "__main__":
    main()
