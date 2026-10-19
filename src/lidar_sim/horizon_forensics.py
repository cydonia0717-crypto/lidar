"""Investigate late completion and parked-vehicle intrusion without changing policy.

All horizons replay the SAME deterministic model and seed. At each stopped
pose we also ask when synthetic CV targets WOULD enter a fixed vehicle's
physical and planner-padded exclusion discs. That counterfactual is *not*
the controller's planned backup path or a safety certificate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from statistics import mean

from .evaluation import TrialSpec, make_trial
from .experiment import RunConfig, run
from .model import State, VehicleConfig
from .offset_sensitivity import true_gap, write_csv


def parse_horizons(value: str) -> tuple[float, ...]:
    try:
        numbers = tuple(float(v.strip()) for v in value.split(","))
    except ValueError as e:
        raise ValueError("horizons must be numeric") from e
    if (len(numbers) < 2 or len(set(numbers)) != len(numbers)
            or any(not math.isfinite(v) or v <= 0 for v in numbers)):
        raise ValueError("horizons must be >=2 distinct finite positive values")
    return tuple(sorted(numbers))


def fixed_pose_intrusion(state: State, world, physical_vehicle: VehicleConfig,
                         *, span_s: float = 6.0, sample_dt: float = .05,
                         planning_padding_m: float = .12) -> dict:
    """Move obstacles under ideal CV while the vehicle is hypothetically fixed.

    Signed actual gap excludes world.safety_margin/padding; negative means
    physical overlap of the conservative target-vs-vehicle covering discs.
    The 'planning threshold' includes both world margin and prediction padding.
    """
    if span_s < 0 or sample_dt <= 0 or planning_padding_m < 0:
        raise ValueError("invalid stationary counterfactual horizon")
    if not world.obstacles:
        return {"minimum_fixed_gap_m": math.inf, "fixed_physical_intrusion_s": None,
                "fixed_planning_envelope_intrusion_s": None}
    n = max(1, math.ceil(span_s / sample_dt))
    minimum = math.inf
    physical_first = None
    envelope_first = None
    center_x, center_y = physical_vehicle.body_center(state)
    threshold = world.safety_margin + planning_padding_m
    for i in range(n + 1):
        relative = min(i * sample_dt, span_s)
        for obs in world.obstacles:
            ox, oy = obs.position(state.time + relative)
            gap = (math.hypot(center_x - ox, center_y - oy)
                   - physical_vehicle.collision_radius - obs.radius)
            minimum = min(minimum, gap)
            if physical_first is None and gap < 0:
                physical_first = relative
            if envelope_first is None and gap < threshold:
                envelope_first = relative
    return {"minimum_fixed_gap_m": minimum,
            "fixed_physical_intrusion_s": physical_first,
            "fixed_planning_envelope_intrusion_s": envelope_first}


def prefix_comparison(baseline_trace: list[dict], longer_trace: list[dict],
                      *, cutoff_s: float) -> dict:
    """Same-scene deterministic prefix check: never infer outcomes from max-time.

    Closed-loop state should be independent of stopping horizon until the
    cutoff. Terminal baseline row omits its unused command; compare states
    only, including that final row when present in the longer trace.
    """
    baseline = {round(float(r["time"]), 6): r for r in baseline_trace
                if float(r["time"]) <= cutoff_s + 1e-6}
    longer = {round(float(r["time"]), 6): r for r in longer_trace
              if float(r["time"]) <= cutoff_s + 1e-6}
    shared = sorted(set(baseline) & set(longer))
    fields = ("x", "y", "yaw", "speed", "steer")
    max_diff = max((abs(float(baseline[t][key]) - float(longer[t][key]))
                    for t in shared for key in fields), default=math.inf)
    missing = sorted(set(baseline) - set(longer))
    return {"shared_frames": len(shared), "missing_baseline_times": len(missing),
            "max_state_difference": max_diff,
            "prefix_equal": bool(shared and not missing and max_diff < 1e-7)}


def annotate_trace(trace: list[dict], world, physical_vehicle: VehicleConfig,
                   *, prediction_padding: float = .12,
                   diagnostic_horizon_s: float = 12.) -> list[dict]:
    data=[]
    for r in trace:
        state=State(float(r["x"]),float(r["y"]),float(r["yaw"]),
                    float(r["speed"]),float(r["steer"]),float(r["time"]))
        gaps=true_gap(state,world,physical_vehicle)
        # The hypothetical parked pose is relevant only while nearly stopped.
        hypothetical = (fixed_pose_intrusion(
            state,world,physical_vehicle,span_s=diagnostic_horizon_s,
            planning_padding_m=prediction_padding)
            if state.speed < .10 else {})
        fields = {key:r.get(key) for key in r
                  if key.startswith("reject_")}
        data.append({
            "time_s":state.time,"x":state.x,"y":state.y,
            "yaw_rad":state.yaw,"speed_mps":state.speed,"steer_rad":state.steer,
            "body_center_x":r.get("body_center_x"),
            "body_center_y":r.get("body_center_y"),
            "abs_heading_error_deg":r.get("abs_heading_error_deg"),
            "command_speed":r.get("command_speed"),
            "command_steer":r.get("command_steer"),
            "applied_speed":r.get("applied_speed"),
            "pending_commands":r.get("pending_commands"),
            "feasible_pairs":r.get("feasible_pairs"),
            "moving_first_action_feasible":r.get("moving_first_action_feasible"),
            "unsafe_fallback":r.get("unsafe_fallback"),
            "fallback_strategy":r.get("fallback_strategy"),
            "emergency":r.get("emergency"),
            "pose_gate_met":r.get("pose_gate_met"),
            "historical_x_line_met":r.get("historical_x_line_met"),
            **gaps, **hypothetical, **fields})
    return data


def windows(frames: list[dict], *, width_s: float = 5.) -> list[dict]:
    if width_s <= 0:
        raise ValueError("window width must be positive")
    if not frames:
        return []
    last = float(frames[-1]["time_s"])
    output=[]
    for index in range(math.ceil(last / width_s)):
        left = index * width_s
        rows = [r for r in frames if left <= float(r["time_s"]) < left + width_s
                and isinstance(r.get("command_speed"), (float,int))]
        if not rows:
            continue
        fea = [r["feasible_pairs"] for r in rows
               if isinstance(r.get("feasible_pairs"), (float,int))]
        stopped = [r for r in rows if r["speed_mps"] < .10]
        envelope = [r["fixed_planning_envelope_intrusion_s"] for r in stopped
                    if r.get("fixed_planning_envelope_intrusion_s") is not None]
        physical = [r["fixed_physical_intrusion_s"] for r in stopped
                    if r.get("fixed_physical_intrusion_s") is not None]
        record = {
            "from_s":left, "to_s":left+width_s,
            "frames":len(rows),"x_displacement_m":rows[-1]["x"]-rows[0]["x"],
            "mean_speed_mps":mean(r["speed_mps"] for r in rows),
            "fraction_stopped":len(stopped)/len(rows),
            "fraction_zero_speed_command":sum(r["command_speed"] < .1 for r in rows)/len(rows),
            "fraction_forward_feasible":sum(bool(r.get("moving_first_action_feasible")) for r in rows)/len(rows),
            "mean_feasible_pairs":mean(fea) if fea else None,
            "min_actual_obstacle_gap_m":min(r["physical_obstacle_gap_m"] for r in rows),
            "min_actual_road_gap_m":min(r["physical_road_gap_m"] for r in rows),
            "stopped_frames_with_planned_envelope_intrusion_12s":len(envelope),
            "stopped_frames_with_physical_intrusion_12s":len(physical),
            "min_time_to_fixed_physical_intrusion_s":min(physical) if physical else None,
        }
        for phase in ("prefix","first","second","brake","hold"):
            for cause in ("road","obstacle"):
                key=f"reject_{phase}_{cause}"
                record[key]=sum(int(r.get(key) or 0) for r in rows)
        output.append(record)
    return output


def study(*, seeds: tuple[int,...], horizons: tuple[float,...],
          assumed_offset_m: float=.55, physical_offset_m: float=.55,
          output: Path) -> dict:
    if not seeds or any(i<0 for i in seeds):
        raise ValueError("need nonnegative scenario seeds")
    if (not math.isfinite(assumed_offset_m) or not math.isfinite(physical_offset_m)
            or not 0<=assumed_offset_m<=.95 or not 0<=physical_offset_m<=.95):
        raise ValueError("invalid offset")
    records=[]
    comparisons=[]
    full_windows=[]
    for seed in seeds:
        trial=make_trial(TrialSpec("two_targets",seed,"weak_brake"))
        vehicle=replace(trial.vehicle,rear_axle_to_body_center_m=assumed_offset_m)
        physical=replace(trial.vehicle,road_footprint="rearaxle_rectangle",
                         rear_axle_to_body_center_m=physical_offset_m)
        simulated={}
        for horizon in horizons:
            result=run("two_targets",seed,"two_stage_bodyframe",vehicle=vehicle,
                       planner=trial.planner,world_override=trial.world,
                       config=RunConfig(max_time=horizon,
                           physical_road_footprint="rearaxle_rectangle",
                           physical_body_offset_m=physical_offset_m,
                           goal_criterion="pose_gate"))
            simulated[horizon]=result
            records.append({
                "seed":seed,"max_time_s":horizon,
                "actual_termination_s":result.duration_s,
                "assumed_offset_m":assumed_offset_m,
                "physical_offset_m":physical_offset_m,
                "success":result.success,"collision":result.collision,
                "timeout":result.timeout,"road_exit":result.road_exit,
                "reached_x_gate":result.reached_historical_x_line,
                "reached_pose_gate":result.reached_pose_gate,
                "final_body_center_y_m":result.final_body_center_y_m,
                "final_heading_error_deg":result.final_abs_heading_error_deg,
                "mean_planning_ms":result.mean_planning_ms})
            frames=annotate_trace(result.trace,trial.world,physical,
                                  prediction_padding=trial.planner.prediction_padding)
            prefix=f"seed{seed}_horizon{horizon:g}"
            write_csv(output/"trajectories"/f"{prefix}.csv",frames)
            for w in windows(frames):
                full_windows.append({"seed":seed,"max_time_s":horizon,**w})
            print(f"seed={seed} horizon={horizon:g}: "
                  f"success={result.success} collision={result.collision} "
                  f"timeout={result.timeout} road_exit={result.road_exit} "
                  f"end={result.duration_s:.2f}s",flush=True)
        shortest=horizons[0]
        for horizon in horizons[1:]:
            if not simulated[shortest].timeout:
                continue
            compare=prefix_comparison(
                simulated[shortest].trace,simulated[horizon].trace,
                cutoff_s=shortest)
            # If either episode ended before baseline's final timestamp, the
            # prefix comparison is a partial overlap, not identity evidence.
            compare["seed"]=seed
            compare["baseline_horizon_s"]=shortest
            compare["extended_horizon_s"]=horizon
            comparisons.append(compare)
    write_csv(output/"horizon_episodes.csv",records)
    write_csv(output/"horizon_prefix_checks.csv",comparisons)
    write_csv(output/"horizon_windows.csv",full_windows)
    if any(not p["prefix_equal"] for p in comparisons):
        raise AssertionError("deterministic prefix changed across simulation horizons")
    report={"mode":"two_stage_bodyframe","scenario":"two_targets",
            "stress":"weak_brake","seeds":seeds,"horizons_s":horizons,
            "assumed_offset_m":assumed_offset_m,
            "physical_offset_m":physical_offset_m,
            "physical_road_footprint":"rearaxle_rectangle",
            "goal_criterion":"pose_gate","goal_lateral_tolerance_m":.75,
            "goal_heading_tolerance_deg":15,
            "padding_m":trial.planner.prediction_padding,
            "interpretation":"synthetic CV truth only; fixed-vehicle forecasts are counterfactual",
            "episodes":records,"prefix_checks":comparisons}
    output.mkdir(parents=True,exist_ok=True)
    (output/"study.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    return report


def main(argv: list[str] | None=None) -> None:
    parser=argparse.ArgumentParser(description="Horizon cutoff and parked intrusion forensics")
    parser.add_argument("--seeds",default="0,1")
    parser.add_argument("--horizons",default="30,35,40")
    parser.add_argument("--assumed-offset",type=float,default=.55)
    parser.add_argument("--physical-offset",type=float,default=.55)
    parser.add_argument("--output",default="results/phase14-forensics")
    args=parser.parse_args(argv)
    payload=study(seeds=tuple(int(s) for s in args.seeds.split(",")),
                  horizons=parse_horizons(args.horizons),
                  assumed_offset_m=args.assumed_offset,
                  physical_offset_m=args.physical_offset,output=Path(args.output))
    print(json.dumps({"episodes":payload["episodes"],
                      "prefix_checks":payload["prefix_checks"]},indent=2),
          flush=True)


if __name__=="__main__":
    main()
