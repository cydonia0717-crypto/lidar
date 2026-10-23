"""Phase18 paired ideal-vs-degraded-sensing closed-loop benchmark.

Unlike the earlier oracle-only run, the planner sees sampled target estimates,
never true target state. The simulated plant and outcomes still use the same
immutable true World. Observations can diverge after trajectories differ
because range/FOV is evaluated at each controller's actual past ego pose.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from .evaluation import TrialSpec, make_trial, VALID_MODES, VALID_SCENARIOS, VALID_STRESSES
from .experiment import RunConfig, run, save_result
from .simulated_perception import PerceptionConfig, SimulatedPerception

PROFILES: dict[str, PerceptionConfig | None] = {
    "oracle": None,
    "clean": PerceptionConfig(
        position_sigma_m=0.0, range_sigma_per_m=0.0,
        dropout_probability=0.0, capture_delay_ticks=0),
    "noisy": PerceptionConfig(),
    "degraded": PerceptionConfig(
        position_sigma_m=.16, range_sigma_per_m=.01,
        dropout_probability=.25, capture_delay_ticks=2,
        max_track_age_s=.35),
}


def run_study(specs: list[TrialSpec], *, modes: tuple[str, ...],
              profiles: tuple[str, ...], max_time: float,
              output: Path) -> list[dict]:
    if not modes or len(set(modes)) != len(modes) or any(
            m not in VALID_MODES for m in modes):
        raise ValueError("invalid or duplicate planner modes")
    if not profiles or len(set(profiles)) != len(profiles) or any(
            p not in PROFILES for p in profiles):
        raise ValueError("invalid or duplicate sensor profiles")
    if not math.isfinite(max_time) or max_time <= 0:
        raise ValueError("max_time must be finite positive")
    if not specs:
        raise ValueError("at least one independent trial required")
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        trial = make_trial(spec)
        # Deterministic observation randomness keyed only by the physical
        # episode, not by the compared planning mode or profile enumeration.
        digest = hashlib.sha256(spec.case_id.encode("utf-8")).digest()
        sample_seed = int.from_bytes(digest[:8], "big")
        for profile in profiles:
            for mode in modes:
                cfg = PROFILES[profile]
                observer = (SimulatedPerception(cfg, sample_seed)
                            if cfg is not None else None)
                result = run(
                    spec.name, spec.seed, mode,
                    vehicle=trial.vehicle, planner=trial.planner,
                    config=RunConfig(
                        max_time=max_time,
                        physical_road_footprint="rearaxle_rectangle",
                        physical_body_offset_m=trial.vehicle.rear_axle_to_body_center_m,
                        goal_criterion="pose_gate",
                        goal_lateral_tolerance_m=.75,
                        goal_heading_tolerance_deg=15.),
                    world_override=trial.world, perception=observer)
                folder = output / spec.case_id / profile / mode
                save_result(result, folder)
                rows.append({
                    "case_id": spec.case_id,
                    "scenario": spec.name,
                    "stress": spec.stress,
                    "seed": spec.seed,
                    "mode": mode,
                    "perception_profile": profile,
                    "observation_seed": sample_seed,
                    **result.metrics(),
                })
    path = output / "episodes.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "inputs.json").write_text(json.dumps({
        "specs": [asdict(s) for s in specs],
        "modes": modes,
        "profiles": {p: (asdict(PROFILES[p]) if PROFILES[p] is not None
                          else "exact oracle target truth") for p in profiles},
        "max_time": max_time,
        "sensor_model": "center detections and causal finite-difference velocities",
        "model_limitations": [
            "no PointCloud2 or beam ray-casting",
            "oracle persistent object association, no ID switching",
            "ground-truth ego pose used by sensor and controller",
            "fixed map corridor and exact known body dimensions",
            "zero sensor processing CPU time; tick-quantized observation lag",
            "no simulated false positive detections, occlusion or semantic classes",
            "brake on unknown scene is not certified safe",
            "physical truth used only for sensing generation and ground-truth outcome",
        ],
        "evaluation": "independent truth plant, read-only estimated-world planner",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Phase18 causal sensor closed-loop ablation")
    p.add_argument("--scenarios", default="crossing,two_targets")
    p.add_argument("--stress", default="nominal")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--modes", default="two_stage_bodyframe,two_stage_viability")
    p.add_argument("--profiles", default="oracle,clean,noisy,degraded")
    p.add_argument("--max-time", type=float, default=30)
    p.add_argument("--output", type=Path,
                   default=Path("results/phase18-sensing-ablation"))
    args = p.parse_args()
    scenarios = tuple(x.strip() for x in args.scenarios.split(","))
    modes = tuple(x.strip() for x in args.modes.split(","))
    profiles = tuple(x.strip() for x in args.profiles.split(","))
    if (not scenarios or any(s not in VALID_SCENARIOS for s in scenarios)
            or args.stress not in VALID_STRESSES
            or args.seeds < 1 or args.seed_start < 0):
        p.error("invalid scenarios/stress/seeds")
    specs = [TrialSpec(scene, seed, args.stress)
             for scene in scenarios
             for seed in range(args.seed_start, args.seed_start+args.seeds)]
    rows = run_study(
        specs, modes=modes, profiles=profiles, max_time=args.max_time,
        output=args.output)
    for r in rows:
        state = ("success" if r["success"] else
                 "collision" if r["collision"] else
                 "road_exit" if r["road_exit"] else "timeout")
        print(f"{r['case_id']} / {r['mode']} / {r['perception_profile']}: "
              f"{state}, blocked={r['perception_blocked_steps']}, "
              f"planned={r['perception_planned_steps']}, "
              f"clearance={r['min_clearance_m']:.3f}m")
    print(f"Saved {len(rows)} truth-separated episodes to {args.output}")


if __name__ == "__main__":
    main()
