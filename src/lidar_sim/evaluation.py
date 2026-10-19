"""Paired, reproducible sensitivity evaluation of Ackermann planning choices.

Every method in one trial sees the SAME target world, simulation duration,
vehicle physics, actuation delay, and initial state. Only the planning logic
differs. All outcomes are synthetic; target motion is supplied as ideal truth.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Iterable

from .experiment import RunConfig, run
from .model import VehicleConfig
from .planner import PlannerConfig
from .world import World, scenario

VALID_MODES = ("reactive", "predictive", "delay_aware", "braking", "braking_hold", "two_stage", "two_stage_rectroad", "two_stage_bodyframe", "two_stage_viability")
VALID_STRESSES = ("nominal", "high_delay", "weak_brake", "fast_targets", "narrow_road")
VALID_SCENARIOS = ("crossing", "two_targets", "late_crossing", "narrow", "empty")


@dataclass(frozen=True)
class TrialSpec:
    name: str
    seed: int
    stress: str

    @property
    def case_id(self) -> str:
        return f"{self.name}__{self.stress}__{self.seed}"


@dataclass(frozen=True)
class Trial:
    spec: TrialSpec
    world: World
    vehicle: VehicleConfig
    planner: PlannerConfig
    target_speed_scale: float


def make_trial(spec: TrialSpec) -> Trial:
    """Independent deterministic episode geometry, shared by all controllers."""
    if spec.name not in VALID_SCENARIOS:
        raise ValueError(f"unsupported scenario: {spec.name}")
    if spec.stress not in VALID_STRESSES:
        raise ValueError(f"unsupported stress: {spec.stress}")
    if spec.seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = random.Random(5011 + spec.seed * 7919 + sum(ord(c) for c in spec.name))
    original = scenario(spec.name, spec.seed)

    # Same sample geometry across ALL stress variants of a given episode.
    adjusted = tuple(replace(obs,
                             x0=obs.x0 + rng.uniform(-.55, .55),
                             y0=obs.y0 + rng.uniform(-.6, .6))
                     for obs in original.obstacles)
    speed_scale = 1.50 if spec.stress == "fast_targets" else 1.0
    adjusted = tuple(replace(obs, vx=obs.vx * speed_scale,
                             vy=obs.vy * speed_scale) for obs in adjusted)
    road_half_width = max(
        1.2, original.road_half_width - (0.50 if spec.stress == "narrow_road" else 0.0))
    world = replace(original, obstacles=adjusted, road_half_width=road_half_width)
    braking = .90 if spec.stress == "weak_brake" else 1.6
    delay = .50 if spec.stress == "high_delay" else .25
    vehicle = VehicleConfig(braking=braking)
    planner = PlannerConfig(control_delay_s=delay)
    return Trial(spec, world, vehicle, planner, speed_scale)


def evaluate_cases(specs: Iterable[TrialSpec], modes: tuple[str, ...] =
                   ("predictive", "delay_aware", "braking"), *,
                   max_time: float = 35.0,
                   physical_road_footprint: str = "disc",
                   goal_criterion: str = "x_gate",
                   goal_lateral_tolerance_m: float = .75,
                   goal_heading_tolerance_deg: float = 15.,
                   assumed_body_offset_m: float | None = None,
                   physical_body_offset_m: float | None = None,
                   failure_trace_dir: str | Path | None = None,
                   failure_traces_per_group: int = 0) -> list[dict]:
    """Return all raw individual rollouts, never aggregate away failures."""
    if not modes or any(mode not in VALID_MODES for mode in modes):
        raise ValueError("choose at least one valid mode")
    if len(set(modes)) != len(modes):
        raise ValueError("duplicate controller modes are not allowed")
    if physical_road_footprint not in ("disc", "rectangle", "rearaxle_rectangle"):
        raise ValueError("unknown physical road footprint")
    if goal_criterion not in ("x_gate", "pose_gate"):
        raise ValueError("invalid goal criterion")
    if not math.isfinite(max_time) or max_time <= 0:
        raise ValueError("max_time must be finite and positive")
    for offset in (assumed_body_offset_m, physical_body_offset_m):
        if offset is not None and (not math.isfinite(offset) or not 0 <= offset <= VehicleConfig().length / 2):
            raise ValueError("invalid rear-axle body offset")
    if failure_traces_per_group < 0:
        raise ValueError("failure trace quota must be nonnegative")
    saved_counts: dict[tuple[str, str, str], int] = {}
    records = []
    for spec in specs:
        trial = make_trial(spec)
        if assumed_body_offset_m is not None:
            trial = replace(
                trial, vehicle=replace(trial.vehicle,
                    rear_axle_to_body_center_m=assumed_body_offset_m))
        for mode in modes:
            outcome = run(spec.name, spec.seed, mode,
                          vehicle=trial.vehicle, planner=trial.planner,
                          config=RunConfig(
                              max_time=max_time,
                              physical_road_footprint=physical_road_footprint,
                              physical_body_offset_m=physical_body_offset_m,
                              goal_criterion=goal_criterion,
                              goal_lateral_tolerance_m=goal_lateral_tolerance_m,
                              goal_heading_tolerance_deg=goal_heading_tolerance_deg),
                          world_override=trial.world)
            trace_file = ""
            if (failure_trace_dir is not None and failure_traces_per_group > 0
                    and (outcome.collision or outcome.road_exit or outcome.timeout)):
                key = (spec.name, spec.stress, mode)
                if saved_counts.get(key, 0) < failure_traces_per_group:
                    folder = Path(failure_trace_dir)
                    folder.mkdir(parents=True, exist_ok=True)
                    target = folder / f"{spec.case_id}__{mode}.csv"
                    _write_csv(target, outcome.trace)
                    trace_file = str(target)
                    saved_counts[key] = saved_counts.get(key, 0) + 1
            records.append({
                "failure_trace_file": trace_file,
                "case_id": spec.case_id, "scenario": spec.name,
                "stress": spec.stress, "seed": spec.seed, "mode": mode,
                "target_speed_scale": trial.target_speed_scale,
                "road_half_width_m": trial.world.road_half_width,
                "physical_road_footprint": physical_road_footprint,
                "physical_body_offset_m": (
                    trial.vehicle.rear_axle_to_body_center_m
                    if physical_body_offset_m is None else physical_body_offset_m),
                "assumed_body_offset_m": trial.vehicle.rear_axle_to_body_center_m,
                "goal_lateral_tolerance_m": goal_lateral_tolerance_m,
                "goal_heading_tolerance_deg": goal_heading_tolerance_deg,
                "control_delay_s": trial.planner.control_delay_s,
                "braking_mps2": trial.vehicle.braking,
                **outcome.metrics(),
            })
    return records


def wilson_95(successes: int, n: int) -> tuple[float, float]:
    """95% Wilson binomial interval; values expressed in [0,1].

    This does NOT account for clustering, dependence across stress variants,
    multiple comparisons, or imperfect physical simulation assumptions.
    """
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("invalid binomial sample")
    z = 1.959963984540054
    p = successes / n
    scale = 1 + z * z / n
    center = (p + z * z / (2 * n)) / scale
    span = z / scale * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # Clamp exact binomial edge cases explicitly. Floating-point rounding
    # may otherwise return 0.9999999999999999 for all successes.
    return (0.0 if successes == 0 else max(0.0, center - span),
            1.0 if successes == n else min(1.0, center + span))


def summarize(records: list[dict]) -> list[dict]:
    """Stratified summaries. Never use failures in mean successful duration."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in records:
        key = row["scenario"], row["stress"], row["mode"]
        groups.setdefault(key, []).append(row)
    summaries = []
    for (name, stress, mode), rows in sorted(groups.items()):
        n = len(rows)
        num_success = sum(bool(r["success"]) for r in rows)
        num_collision = sum(bool(r["collision"]) for r in rows)
        num_timeout = sum(bool(r.get("timeout", False)) for r in rows)
        num_road_exit = sum(bool(r.get("road_exit", False)) for r in rows)
        success_low, success_high = wilson_95(num_success, n)
        collision_low, collision_high = wilson_95(num_collision, n)
        good_times = [r["duration_s"] for r in rows if r["success"]]
        summaries.append({
            "scenario": name, "stress": stress, "mode": mode,
            "n": n, "successes": num_success, "collisions": num_collision,
            "timeouts": num_timeout, "road_exits": num_road_exit,
            "success_rate": num_success / n, "success_ci_low": success_low,
            "success_ci_high": success_high, "collision_rate": num_collision / n,
            "collision_ci_low": collision_low, "collision_ci_high": collision_high,
            "mean_success_time_s": mean(good_times) if good_times else "",
            "mean_min_clearance_m": mean(r["min_clearance_m"] for r in rows),
            "mean_stopped_steps": mean(r["stopped_steps"] for r in rows),
            "mean_unsafe_fallback_steps": mean(r["unsafe_fallback_steps"] for r in rows),
            "mean_backup_rejections": mean(r["backup_rejections"] for r in rows),
            "mean_planning_ms": mean(r["mean_planning_ms"] for r in rows),
        })
    return summaries


def paired_braking_effects(records: list[dict]) -> list[dict]:
    """Within-episode paired braking versus delay-aware decision differences."""
    trials: dict[str, dict[str, dict]] = {}
    for row in records:
        slots = trials.setdefault(row["case_id"], {})
        if row["mode"] in slots:
            raise ValueError("duplicate case/mode result")
        slots[row["mode"]] = row
    pairs = []
    for case_id, methods in sorted(trials.items()):
        if "braking" not in methods or "delay_aware" not in methods:
            continue
        a = methods["delay_aware"]
        b = methods["braking"]
        keys = ("scenario", "stress", "seed", "target_speed_scale",
                "road_half_width_m", "control_delay_s", "braking_mps2")
        if any(a[k] != b[k] for k in keys):
            raise ValueError(f"unpaired physics or scenario for {case_id}")
        pairs.append({
            "case_id": case_id, "scenario": a["scenario"],
            "stress": a["stress"], "seed": a["seed"],
            "delay_aware_success": bool(a["success"]), "braking_success": bool(b["success"]),
            "success_change": int(bool(b["success"])) - int(bool(a["success"])),
            "delay_aware_collision": bool(a["collision"]), "braking_collision": bool(b["collision"]),
            "collision_change": int(bool(b["collision"])) - int(bool(a["collision"])),
            "clearance_change_m": b["min_clearance_m"] - a["min_clearance_m"],
            "stopped_steps_change": b["stopped_steps"] - a["stopped_steps"],
            "braking_unsafe_fallback_steps": b["unsafe_fallback_steps"],
            # A duration difference is interpretable only when both reach goal.
            "success_time_change_s":
                b["duration_s"] - a["duration_s"]
                if b["success"] and a["success"] else "",
        })
    return pairs


def paired_post_stop_effects(records: list[dict]) -> list[dict]:
    """Compare post-stop occupancy filtering to legacy braking by SAME trial.

    Success, collision, timeout and road exit must remain separate outcomes.
    Results cannot be interpreted as improved safety without considering
    changes in completion probability and travel time.
    """
    trials: dict[str, dict[str, dict]] = {}
    for row in records:
        bucket = trials.setdefault(row["case_id"], {})
        if row["mode"] in bucket:
            raise ValueError("duplicate case/mode result")
        bucket[row["mode"]] = row
    pairs = []
    for key, methods in sorted(trials.items()):
        if "braking" not in methods or "braking_hold" not in methods:
            continue
        a = methods["braking"]
        b = methods["braking_hold"]
        same = ("scenario", "stress", "seed", "target_speed_scale",
                "road_half_width_m", "control_delay_s", "braking_mps2")
        if any(a[k] != b[k] for k in same):
            raise ValueError(f"unpaired physical trial: {key}")
        pair = {
            "case_id": key, "scenario": a["scenario"],
            "stress": a["stress"], "seed": a["seed"],
        }
        for metric in ("success", "collision", "timeout", "road_exit"):
            pair["braking_" + metric] = bool(a[metric])
            pair["braking_hold_" + metric] = bool(b[metric])
            pair[metric + "_change"] = int(bool(b[metric])) - int(bool(a[metric]))
        pair["clearance_change_m"] = b["min_clearance_m"] - a["min_clearance_m"]
        pair["stopped_steps_change"] = b["stopped_steps"] - a["stopped_steps"]
        pair["hold_backup_rejections"] = b["backup_rejections"]
        pair["hold_unsafe_fallback_steps"] = b["unsafe_fallback_steps"]
        pair["both_success_time_change_s"] = (
            b["duration_s"] - a["duration_s"]
            if b["success"] and a["success"] else "")
        pairs.append(pair)
    return pairs



def paired_two_stage_effects(records: list[dict]) -> list[dict]:
    """Paired decision-structure comparison, including computation overhead."""
    cases: dict[str, dict[str, dict]] = {}
    for row in records:
        dst = cases.setdefault(row["case_id"], {})
        if row["mode"] in dst:
            raise ValueError("duplicate case/mode")
        dst[row["mode"]] = row
    result = []
    for case_id, modes in sorted(cases.items()):
        if not {"braking_hold", "two_stage"}.issubset(modes):
            continue
        old, new = modes["braking_hold"], modes["two_stage"]
        for key in ("scenario", "stress", "seed", "target_speed_scale",
                    "road_half_width_m", "control_delay_s", "braking_mps2"):
            if old[key] != new[key]:
                raise ValueError("nonpaired world/physics " + case_id)
        values = {"case_id": case_id, "scenario": old["scenario"],
                  "stress": old["stress"], "seed": old["seed"]}
        for metric in ("success", "collision", "timeout", "road_exit"):
            values["hold_" + metric] = bool(old[metric])
            values["two_stage_" + metric] = bool(new[metric])
            values[metric + "_change"] = int(bool(new[metric])) - int(bool(old[metric]))
        values["hold_mean_planning_ms"] = old["mean_planning_ms"]
        values["two_stage_mean_planning_ms"] = new["mean_planning_ms"]
        values["planning_ms_change"] = new["mean_planning_ms"] - old["mean_planning_ms"]
        values["both_success_time_change_s"] = (
            new["duration_s"] - old["duration_s"]
            if old["success"] and new["success"] else "")
        result.append(values)
    return result



def paired_road_footprint_effects(records: list[dict]) -> list[dict]:
    """Disc-planner vs actual rotated-body planner under SAME physical truth.

    This deliberately requires identical physical collision rules for both
    methods; a false improvement obtained by changing ground-truth criteria
    would be invalid.
    """
    groups: dict[str, dict[str, dict]] = {}
    for r in records:
        dest = groups.setdefault(r["case_id"], {})
        if r["mode"] in dest:
            raise ValueError("duplicate case/mode")
        dest[r["mode"]] = r
    result = []
    for case_id, rows in sorted(groups.items()):
        if not {"two_stage", "two_stage_rectroad"}.issubset(rows):
            continue
        baseline, actual = rows["two_stage"], rows["two_stage_rectroad"]
        keys = ("scenario", "stress", "seed", "target_speed_scale",
                "road_half_width_m", "control_delay_s", "braking_mps2",
                "physical_road_footprint")
        if any(baseline[k] != actual[k] for k in keys):
            raise ValueError("unmatched physical road truth or world: " + case_id)
        if baseline["physical_road_footprint"] != "rectangle":
            raise ValueError("geometry ablation requires common oriented rectangle physical truth")
        pair={"case_id": case_id, "scenario": baseline["scenario"],
              "stress": baseline["stress"], "seed": baseline["seed"],
              "physical_road_footprint": baseline["physical_road_footprint"]}
        for key in ("success","collision","road_exit","timeout"):
            pair["disc_"+key]=bool(baseline[key])
            pair["rectangle_"+key]=bool(actual[key])
            pair[key+"_change"]=int(bool(actual[key]))-int(bool(baseline[key]))
        pair["planning_ms_change"]=actual["mean_planning_ms"]-baseline["mean_planning_ms"]
        pair["both_success_time_change_s"]=(
            actual["duration_s"]-baseline["duration_s"]
            if actual["success"] and baseline["success"] else "")
        result.append(pair)
    return result



def paired_bodyframe_effects(records: list[dict]) -> list[dict]:
    """True rear-axle physical body frame with shared pose-gate semantics."""
    groups: dict[str, dict[str, dict]] = {}
    for r in records:
        slot = groups.setdefault(r["case_id"], {})
        if r["mode"] in slot:
            raise ValueError("duplicate case/mode")
        slot[r["mode"]] = r
    result = []
    for key, modes in sorted(groups.items()):
        if not {"two_stage_rectroad", "two_stage_bodyframe"}.issubset(modes):
            continue
        a, b = modes["two_stage_rectroad"], modes["two_stage_bodyframe"]
        fields = ("scenario", "stress", "seed", "target_speed_scale",
                  "road_half_width_m", "control_delay_s", "braking_mps2",
                  "physical_road_footprint", "goal_criterion",
                  "goal_lateral_tolerance_m", "goal_heading_tolerance_deg")
        if any(a[k] != b[k] for k in fields):
            raise ValueError("unpaired physical world, reference frame or goal policy")
        if a["physical_road_footprint"] != "rearaxle_rectangle":
            raise ValueError("frame ablation requires common rear axle physical truth")
        if a["goal_criterion"] != "pose_gate":
            raise ValueError("frame ablation requires common pose gate")
        out={"case_id":key,"scenario":a["scenario"],"stress":a["stress"],
             "seed":a["seed"]}
        for field in ("success","collision","timeout","road_exit",
                      "reached_historical_x_line","reached_pose_gate"):
            out["center_"+field]=bool(a[field])
            out["rearaxle_"+field]=bool(b[field])
            out[field+"_change"]=int(bool(b[field]))-int(bool(a[field]))
        out["planning_ms_change"]=b["mean_planning_ms"]-a["mean_planning_ms"]
        out["both_success_time_change_s"]=(
            b["duration_s"]-a["duration_s"] if a["success"] and b["success"] else "")
        result.append(out)
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def save_study(records: list[dict], output: str | Path,
               *, inputs: dict | None = None, plot: bool = False) -> dict[str, Path]:
    """Write exactly the calculated run-level rows and reproducibility metadata."""
    if not records:
        raise ValueError("no evaluation records")
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    files = {
        "episodes": folder / "episodes.csv",
        "summary": folder / "summary_by_scenario_stress_mode.csv",
        "paired": folder / "paired_braking_effects.csv",
        "paired_post_stop": folder / "paired_post_stop_effects.csv",
        "paired_two_stage": folder / "paired_two_stage_effects.csv",
        "paired_road_footprint": folder / "paired_road_footprint_effects.csv",
        "paired_bodyframe": folder / "paired_bodyframe_effects.csv",
        "metadata": folder / "metadata.json",
    }
    _write_csv(files["episodes"], records)
    _write_csv(files["summary"], summarize(records))
    pairs = paired_braking_effects(records)
    if pairs:
        _write_csv(files["paired"], pairs)
    else:
        files.pop("paired")
    hold_pairs = paired_post_stop_effects(records)
    if hold_pairs:
        _write_csv(files["paired_post_stop"], hold_pairs)
    else:
        files.pop("paired_post_stop")
    stage_pairs = paired_two_stage_effects(records)
    if stage_pairs:
        _write_csv(files["paired_two_stage"], stage_pairs)
    else:
        files.pop("paired_two_stage")
    footprint_pairs = paired_road_footprint_effects(records)
    if footprint_pairs:
        _write_csv(files["paired_road_footprint"], footprint_pairs)
    else:
        files.pop("paired_road_footprint")
    frame_pairs = paired_bodyframe_effects(records)
    if frame_pairs:
        _write_csv(files["paired_bodyframe"], frame_pairs)
    else:
        files.pop("paired_bodyframe")
    payload = {
        "source": "synthetic 2D Ackermann world, ideal CV obstacle motion truth",
        "not": "real LiDAR data, Kalman estimates, or a safety guarantee",
        "actual_execution_delay": "identical within a paired case for all modes",
        "baseline_padding": "identical cfg.prediction_padding for all planner modes",
        "success_condition": "ego x reaches goal_x - goal_tolerance; yaw/lateral terminal error not enforced",
        "collision_check": "discrete sampled conservative discs; no continuous-time guarantee",
        "confidence_interval": "per-stratum 95% Wilson interval; excludes cluster dependence",
        "design": inputs or {},
        "n_episodes": len(records),
        "n_distinct_trials": len({r["case_id"] for r in records}),
    }
    files["metadata"].write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    if plot:
        files["figure"] = folder / "rates_by_stress.png"
        plot_study(records, files["figure"])
    return files


def plot_study(records: list[dict], path: str | Path) -> None:
    """Render two metric panels directly from run-level Boolean outcomes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strata = sorted({r["stress"] for r in records})
    modes = [m for m in VALID_MODES if any(r["mode"] == m for r in records)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, outcome_key, label in zip(
            axes, ("success", "collision"), ("Task success", "Physical collision")):
        for i, mode in enumerate(modes):
            y, lo, hi = [], [], []
            for stress in strata:
                subset = [r for r in records
                          if r["stress"] == stress and r["mode"] == mode]
                n = len(subset)
                if not n:
                    y.append(float("nan"))
                    lo.append(float("nan"))
                    hi.append(float("nan"))
                    continue
                successes = sum(bool(r[outcome_key]) for r in subset)
                low, high = wilson_95(successes, n)
                y.append(successes / n * 100)
                lo.append(low * 100)
                hi.append(high * 100)
            x = [j + (i - (len(modes) - 1)/2) * .17 for j in range(len(strata))]
            ax.errorbar(x, y,
                        yerr=[[v - low for v, low in zip(y, lo)],
                              [high - v for high, v in zip(hi, y)]],
                        marker="o", linewidth=1.4, capsize=3,
                        label=mode, color=colors[i % len(colors)])
        ax.set_xticks(list(range(len(strata))), strata, rotation=20, ha="right")
        ax.set_ylim(0, 104)
        ax.set_ylabel("Episode rate (%)")
        ax.set_title(label + " (95% Wilson interval)")
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.suptitle("Synthetic simulation: paired scenarios, identical plant physics",
                 fontsize=13)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def parse_names(arg: str, valid: tuple[str, ...], label: str) -> tuple[str, ...]:
    values = tuple(v.strip() for v in arg.split(",") if v.strip())
    if not values or len(values) != len(set(values)) or any(v not in valid for v in values):
        raise ValueError(f"invalid {label}: choose comma-separated names from {valid}")
    return values


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Paired simulated Ackermann decision evaluation")
    parser.add_argument("--scenarios", default="crossing,two_targets,late_crossing,narrow")
    parser.add_argument("--stresses", default="nominal,high_delay,weak_brake,fast_targets,narrow_road")
    parser.add_argument("--modes", default="predictive,delay_aware,braking")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--goal-criterion", choices=("x_gate", "pose_gate"),
                        default="x_gate")
    parser.add_argument("--goal-lateral-tolerance", type=float, default=.75)
    parser.add_argument("--goal-heading-tolerance-deg", type=float, default=15.)
    parser.add_argument("--seed-start", type=int, default=0,
                        help="Disjoint held-out seed window")
    parser.add_argument("--physical-road-footprint", choices=("disc", "rectangle", "rearaxle_rectangle"),
                        default="disc",
                        help="Common physical ground truth for ALL planner modes")
    parser.add_argument("--max-time", type=float, default=35.0)
    parser.add_argument("--output", default="results/evaluation")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--print-episodes", action="store_true",
                        help="Print each episode status to diagnose non-collision failures")
    parser.add_argument("--failure-traces-per-group", type=int, default=2,
                        help="Save up to N collision/timeout traces per scenario/stress/mode")
    args = parser.parse_args(argv)
    if (args.seeds < 1 or args.seed_start < 0 or not math.isfinite(args.max_time) or args.max_time <= 0
            or args.failure_traces_per_group < 0):
        parser.error("seeds/max-time must be positive and trace quota nonnegative")
    try:
        names = parse_names(args.scenarios, VALID_SCENARIOS, "scenarios")
        stresses = parse_names(args.stresses, VALID_STRESSES, "stresses")
        modes = parse_names(args.modes, VALID_MODES, "modes")
    except ValueError as exc:
        parser.error(str(exc))
    specs = [TrialSpec(name, seed, stress) for name in names
             for stress in stresses for seed in range(args.seed_start, args.seed_start+args.seeds)]
    print(f"Running {len(specs)} paired trials x {len(modes)} planner modes = "
          f"{len(specs) * len(modes)} synthetic rollouts", flush=True)
    results = evaluate_cases(
        specs, modes, max_time=args.max_time,
        physical_road_footprint=args.physical_road_footprint,
        goal_criterion=args.goal_criterion,
        goal_lateral_tolerance_m=args.goal_lateral_tolerance,
        goal_heading_tolerance_deg=args.goal_heading_tolerance_deg,
                             failure_trace_dir=Path(args.output) / "failure_traces",
                             failure_traces_per_group=args.failure_traces_per_group)
    files = save_study(results, args.output,
                       inputs={"scenarios": names, "stresses": stresses, "modes": modes,
                               "seeds_per_stratum": args.seeds,
                               "seed_start": args.seed_start,
                               "max_time_s": args.max_time,
                               "physical_road_footprint": args.physical_road_footprint,
                               "goal_criterion": args.goal_criterion,
                               "goal_lateral_tolerance_m": args.goal_lateral_tolerance,
                               "goal_heading_tolerance_deg": args.goal_heading_tolerance_deg,
                               "max_failure_traces_per_group": args.failure_traces_per_group},
                       plot=args.plot)
    for name, path in files.items():
        print(f"{name}: {path}")
    print("saved_failure_traces:", sum(bool(row["failure_trace_file"]) for row in results))
    for row in summarize(results):
        print(f"{row['scenario']}/{row['stress']}/{row['mode']}: "
              f"success={row['successes']}/{row['n']}, "
              f"collision={row['collisions']}/{row['n']}, "
              f"timeout={row['timeouts']}/{row['n']}, "
              f"road_exit={row['road_exits']}/{row['n']}, "
              f"mean_success_time={row['mean_success_time_s']}, "
              f"mean_planning_ms={row['mean_planning_ms']:.3f}")
    if args.print_episodes:
        for row in results:
            print(f"EPISODE {row['case_id']} {row['mode']}: "
                  f"success={row['success']}, collision={row['collision']}, "
                  f"timeout={row['timeout']}, road_exit={row['road_exit']}, "
                  f"duration={row['duration_s']:.2f}s, "
                  f"fallback_steps={row['unsafe_fallback_steps']}, "
                  f"backup_rejections={row['backup_rejections']}")


if __name__ == "__main__":
    main()
