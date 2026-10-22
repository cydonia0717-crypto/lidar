"""Strict phase15-vs-phase16 closed-loop replay and timing comparison.

This is a *no-compute-delay* synthetic simulation equivalence audit.
Reference is frozen PR #15 code, not the two_stage_bodyframe baseline.
A different wall-clock planning time must not fail deterministic replay.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .phase16_timing import summarize

OLD = "two_stage_viability_phase15"
NEW = "two_stage_viability"
# Include issued commands, applied commands and state; metadata/cost_parts can
# legitimately differ because phase16 deliberately skips unnecessary probes.
PHYSICAL_KEYS = (
    "time", "x", "y", "yaw", "speed", "steer", "min_clearance",
    "body_center_x", "body_center_y", "abs_heading_error_deg",
    "historical_x_line_met", "pose_gate_met", "command_speed",
    "command_steer", "applied_speed", "applied_steer",
    "pending_commands", "unsafe_fallback", "fallback_strategy", "emergency",
    "feasible_pairs", "backup_rejections",
)
OUTCOME_KEYS = (
    "success", "collision", "timeout", "road_exit", "duration_s",
    "min_clearance_m", "emergency_steps", "unsafe_fallback_steps",
)


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_case(old_trace: Path, new_trace: Path) -> dict:
    old_rows, new_rows = load_csv(old_trace), load_csv(new_trace)
    if len(old_rows) != len(new_rows):
        raise AssertionError(
            f"{old_trace.name}: frame count changed: {len(old_rows)} -> {len(new_rows)}")
    for i, (a, b) in enumerate(zip(old_rows, new_rows)):
        for key in PHYSICAL_KEYS:
            if key not in a or key not in b:
                raise AssertionError(f"{old_trace.name}: missing {key}")
            if a[key] != b[key]:
                raise AssertionError(
                    f"{old_trace.name}: first divergence at frame {i}, "
                    f"t={a['time']}: {key}={a[key]!r} vs {b[key]!r}")
    old_t = summarize(old_trace)
    new_t = summarize(new_trace)
    return {
        "case_id": old_trace.name.rsplit("__", 1)[0],
        "equivalent_frames": len(old_rows),
        "phase15_mean_ms": old_t["mean_ms"],
        "phase16_mean_ms": new_t["mean_ms"],
        "speedup_ratio": old_t["mean_ms"] / new_t["mean_ms"],
        "phase15_p95_ms": old_t["p95_ms"],
        "phase16_p95_ms": new_t["p95_ms"],
        "phase15_p99_ms": old_t["p99_ms"],
        "phase16_p99_ms": new_t["p99_ms"],
        "phase15_over100_fraction": old_t["over_period_fraction"],
        "phase16_over100_fraction": new_t["over_period_fraction"],
        "phase15_decisions": old_t["decisions"],
        "phase16_decisions": new_t["decisions"],
    }


def verify_root(root: Path) -> list[dict]:
    trace_dir = root / "all_traces"
    originals = sorted(trace_dir.glob(f"*__{OLD}.csv"))
    optimized = sorted(trace_dir.glob(f"*__{NEW}.csv"))
    if not originals or len(originals) != len(optimized):
        raise ValueError(
            f"{root}: unpaired phase15/phase16 trace counts "
            f"{len(originals)} vs {len(optimized)}")
    episode_rows = load_csv(root / "episodes.csv")
    by_case: dict[str, dict] = {}
    for row in episode_rows:
        if row["mode"] in (OLD, NEW):
            key = row["case_id"]
            mode_rows = by_case.setdefault(key, {})
            if row["mode"] in mode_rows:
                raise AssertionError(f"duplicate mode: {key}")
            mode_rows[row["mode"]] = row
    if set(by_case) != {p.name.removesuffix(f"__{OLD}.csv") for p in originals}:
        raise AssertionError("episodes.csv and full traces cover different cases")
    result = []
    for path in originals:
        pair = path.with_name(path.name.replace(f"__{OLD}.csv", f"__{NEW}.csv"))
        if not pair.exists():
            raise ValueError(f"missing paired trace: {pair}")
        key = path.name.removesuffix(f"__{OLD}.csv")
        versions = by_case[key]
        if set(versions) != {OLD, NEW}:
            raise AssertionError(f"unpaired episode result: {key}")
        for outcome in OUTCOME_KEYS:
            if versions[OLD][outcome] != versions[NEW][outcome]:
                raise AssertionError(
                    f"{key}: {outcome} differs: "
                    f"{versions[OLD][outcome]} vs {versions[NEW][outcome]}")
        # Trial metadata must be matched, not silently compare altered physics.
        for field in (
            "road_half_width_m", "control_delay_s", "braking_mps2",
            "physical_road_footprint", "physical_body_offset_m",
            "assumed_body_offset_m", "goal_criterion",
            "goal_lateral_tolerance_m", "goal_heading_tolerance_deg",
        ):
            if versions[OLD][field] != versions[NEW][field]:
                raise AssertionError(f"{key}: mismatched trial config: {field}")
        result.append(verify_case(path, pair))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("results/phase16-equivalence.csv"))
    args = parser.parse_args()
    rows = [item for root in args.roots for item in verify_root(root)]
    if not rows:
        parser.error("no paired episodes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['case_id']}: identical {r['equivalent_frames']} frames; "
              f"phase15={r['phase15_mean_ms']:.1f}ms "
              f"phase16={r['phase16_mean_ms']:.1f}ms "
              f"speedup={r['speedup_ratio']:.2f}x "
              f"phase16 P99={r['phase16_p99_ms']:.1f}ms "
              f"over100={r['phase16_over100_fraction']:.1%}")
    print(f"All {len(rows)} matched reference-vs-optimized episodes are equivalent.")


if __name__ == "__main__":
    main()
