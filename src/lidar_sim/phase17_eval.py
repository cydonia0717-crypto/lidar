"""Phase17: run fixed-vs-measured compute-lag closed loops.

All modes use the same synthetic world and motor dynamics as phase16. The
planner's measured compute time is fed into submission delay only when the
experimental timing policy requests it. Physical actuation still adds the
original command-transport latency (0.25s nominal).
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from .evaluation import TrialSpec, make_trial
from .experiment import RunConfig, run, save_result
from .phase16_timing import percentile


def parse_policies(text: str) -> list[tuple[str, float]]:
    policies = []
    for token in text.split(","):
        token = token.strip()
        if token in ("none", "measured"):
            policies.append((token, 0.0))
        elif token.startswith("fixed:"):
            value = float(token.split(":", 1)[1])
            if not math.isfinite(value) or value < 0:
                raise ValueError("fixed compute ms must be finite and nonnegative")
            policies.append(("fixed", value))
        else:
            raise ValueError(f"unknown compute policy: {token}")
    if not policies or len(policies) != len(set(policies)):
        raise ValueError("nonempty distinct policies required")
    return policies


def summarize_timing(trace: list[dict]) -> dict:
    values = [float(r["planning_ms"]) for r in trace
              if r.get("planning_ms") not in ("", None)]
    if not values:
        raise ValueError("no planning timing samples")
    return {
        "n_plans": len(values),
        "mean_planning_ms": sum(values) / len(values),
        "p95_planning_ms": percentile(values, 95),
        "p99_planning_ms": percentile(values, 99),
        "max_planning_ms": max(values),
        "over_100ms_count": sum(t > 100 for t in values),
        "over_100ms_fraction": sum(t > 100 for t in values) / len(values),
    }


def execute(spec: TrialSpec, mode: str, policy: str, fixed_ms: float,
            max_time: float, output: Path, scale: float = 1.0) -> dict:
    trial = make_trial(spec)
    rcfg = RunConfig(
        max_time=max_time,
        physical_road_footprint="rearaxle_rectangle",
        physical_body_offset_m=.55,
        goal_criterion="pose_gate",
        compute_delay_mode=policy,
        compute_fixed_ms=fixed_ms,
        compute_delay_scale=scale,
    )
    result = run(
        spec.name, spec.seed, mode, vehicle=trial.vehicle,
        planner=trial.planner, world_override=trial.world, config=rcfg)
    label = policy if policy != "fixed" else f"fixed_{fixed_ms:g}ms"
    save_result(result, output / spec.case_id / mode / label)
    return {
        "case_id": spec.case_id,
        "mode": mode,
        "compute_policy": policy,
        "fixed_compute_ms": fixed_ms,
        "compute_scale": scale,
        "success": result.success,
        "collision": result.collision,
        "road_exit": result.road_exit,
        "timeout": result.timeout,
        "duration_s": result.duration_s,
        "min_clearance_m": result.min_clearance_m,
        "missed_control_ticks": result.missed_control_ticks,
        "submitted_commands": result.submitted_commands,
        **summarize_timing(result.trace),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase17 latency-feedback counterfactual, NOT realtime certification")
    parser.add_argument("--scenarios", default="two_targets")
    parser.add_argument("--stress", default="weak_brake")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--modes", default="two_stage_bodyframe,two_stage_viability")
    parser.add_argument("--policies", default="none,fixed:80,fixed:160,measured")
    parser.add_argument("--max-time", type=float, default=35.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("results/phase17"))
    args = parser.parse_args()
    policies = parse_policies(args.policies)
    if args.seed_start < 0 or args.seeds < 1:
        parser.error("invalid seed window")
    if not math.isfinite(args.max_time) or args.max_time <= 0:
        parser.error("invalid time horizon")
    if not math.isfinite(args.scale) or args.scale <= 0:
        parser.error("compute scale must be finite and positive")
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    modes = [s.strip() for s in args.modes.split(",") if s.strip()]
    records = []
    for scene in scenarios:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            spec = TrialSpec(scene, seed, args.stress)
            for mode in modes:
                for policy, fixed_ms in policies:
                    row = execute(spec, mode, policy, fixed_ms,
                                  args.max_time, args.output, args.scale)
                    records.append(row)
                    print(f"{scene}/{seed} {mode} {policy}:{fixed_ms:g}ms "
                          f"success={row['success']} collision={row['collision']} "
                          f"skipped={row['missed_control_ticks']} "
                          f"p99={row['p99_planning_ms']:.1f}ms", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} complete phase17 rollouts.")


if __name__ == "__main__":
    main()
