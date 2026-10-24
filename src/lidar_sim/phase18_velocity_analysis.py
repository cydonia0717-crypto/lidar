"""Matched synthetic velocity-filter ablation; no safety guarantee from N cases.

Compares episode-level outcomes and *only scored* predicted-world errors. The
conditioned errors are never substituted for dropout or blocked time steps.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


PAIRS = (
    ("noisy", "noisy_ls5"),
    ("degraded", "degraded_ls5"),
)
ERROR_KEYS = (
    "perception_eval_mean_position_error_m",
    "perception_eval_mean_velocity_error_mps",
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bool_value(value: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError("bad outcome bool: "+repr(value))
    return value == "True"


def scored_errors(path: Path) -> dict[str, float | int | str]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"empty trace: {path}")
    scored = []
    for row in rows:
        # Terminal rows do not represent a completed sensing/planning tick.
        if row.get("perception_source") != "causal_estimated_tracks":
            continue
        scored.append(row)
    results: dict[str, float | int | str] = {
        "decision_ticks": len(scored),
    }
    for key in ERROR_KEYS:
        data = []
        for row in scored:
            value = row.get(key, "")
            if value in ("", None):
                continue
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"invalid scored error {key}: {value}")
            data.append(number)
        results[key+"_scored_ticks"] = len(data)
        results[key+"_mean_over_scored_ticks"] = (
            sum(data) / len(data) if data else "")
    return results


def evaluate(root: Path, *,
             pairs: tuple[tuple[str, str], ...] = PAIRS) -> list[dict]:
    records = read_csv(root/"episodes.csv")
    if not records:
        raise ValueError("no episodes")
    by_id = {}
    for row in records:
        key = (row["case_id"], row["mode"], row["perception_profile"])
        if key in by_id:
            raise ValueError("duplicated case/mode/profile: "+repr(key))
        by_id[key] = row
    output = []
    for case_id, mode, profile in sorted(by_id):
        pair = next(((reference, candidate)
                     for reference, candidate in pairs if profile == reference),
                    None)
        if pair is None:
            continue
        old, new = pair
        before = by_id[case_id, mode, old]
        other_key = (case_id, mode, new)
        if other_key not in by_id:
            raise ValueError("missing paired estimator: "+repr(other_key))
        after = by_id[other_key]
        if before["observation_seed"] != after["observation_seed"]:
            raise ValueError("different synthetic observation seeds")
        if (before["scenario"], before["stress"], before["seed"]) != (
                after["scenario"], after["stress"], after["seed"]):
            raise ValueError("different physical cases")
        if not (bool_value(before["perception_enabled"])
                and bool_value(after["perception_enabled"])):
            raise ValueError("oracle profile is not a velocity-filter pair")
        reports = []
        for source, row in ((old, before), (new, after)):
            trace_path = (
                root / case_id / source / mode /
                f"{row['scenario']}_{mode}_{row['seed']}_trace.csv")
            reports.append(scored_errors(trace_path))
        before_err, after_err = reports
        result = {
            "case_id": case_id,
            "scenario": before["scenario"],
            "stress": before["stress"],
            "seed": before["seed"],
            "mode": mode,
            "reference_profile": old,
            "candidate_profile": new,
            "observation_seed": before["observation_seed"],
            "reference_success": bool_value(before["success"]),
            "candidate_success": bool_value(after["success"]),
            "reference_collision": bool_value(before["collision"]),
            "candidate_collision": bool_value(after["collision"]),
            "reference_road_exit": bool_value(before["road_exit"]),
            "candidate_road_exit": bool_value(after["road_exit"]),
            "reference_timeout": bool_value(before["timeout"]),
            "candidate_timeout": bool_value(after["timeout"]),
            "reference_duration_s": float(before["duration_s"]),
            "candidate_duration_s": float(after["duration_s"]),
            "reference_min_clearance_m": float(before["min_clearance_m"]),
            "candidate_min_clearance_m": float(after["min_clearance_m"]),
            "reference_perception_blocked_steps": int(
                before["perception_blocked_steps"]),
            "candidate_perception_blocked_steps": int(
                after["perception_blocked_steps"]),
            "reference_perception_planned_steps": int(
                before["perception_planned_steps"]),
            "candidate_perception_planned_steps": int(
                after["perception_planned_steps"]),
        }
        for key in ("decision_ticks", *(k+"_scored_ticks" for k in ERROR_KEYS),
                    *(k+"_mean_over_scored_ticks" for k in ERROR_KEYS)):
            result["reference_"+key] = before_err[key]
            result["candidate_"+key] = after_err[key]
        result["collision_changed"] = result[
            "reference_collision"] != result["candidate_collision"]
        result["success_changed"] = result[
            "reference_success"] != result["candidate_success"]
        output.append(result)
    if not output:
        raise ValueError("no estimator pairs")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired causal CV velocity estimator diagnostics")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("results/phase18-velocity-paired.csv"))
    args = parser.parse_args()
    rows = evaluate(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['case_id']} {r['mode']} "
              f"{r['reference_profile']} -> {r['candidate_profile']}: "
              f"collision={int(r['reference_collision'])}"
              f"->{int(r['candidate_collision'])}; "
              f"success={int(r['reference_success'])}"
              f"->{int(r['candidate_success'])}; "
              "true-velocity errors are offline scored-only measurements")
    print(f"{len(rows)} matched pairs, not a statistical or physical safety guarantee.")


if __name__ == "__main__":
    main()
