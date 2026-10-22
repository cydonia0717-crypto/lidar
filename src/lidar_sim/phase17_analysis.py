"""Phase17 paired interpretation: control outcomes without/with CPU-lag feedback."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def pair_rows(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for row in records:
        if row["compute_policy"] not in ("none", "measured"):
            continue
        key = (row["case_id"], row["mode"])
        slots = grouped.setdefault(key, {})
        policy = row["compute_policy"]
        if policy in slots:
            raise ValueError(f"duplicate policy {policy}: {key}")
        slots[policy] = row
    results = []
    for (case_id, mode), policies in sorted(grouped.items()):
        if set(policies) != {"none", "measured"}:
            raise ValueError(f"missing matched measured/none pair: {case_id}/{mode}")
        a, b = policies["none"], policies["measured"]
        for field in ("fixed_compute_ms", "compute_scale"):
            if a[field] != b[field]:
                raise ValueError(f"{case_id}/{mode}: mismatched trial {field}")
        results.append({
            "case_id": case_id, "mode": mode,
            "none_success": a["success"], "measured_success": b["success"],
            "none_collision": a["collision"], "measured_collision": b["collision"],
            "none_timeout": a["timeout"], "measured_timeout": b["timeout"],
            "none_road_exit": a["road_exit"], "measured_road_exit": b["road_exit"],
            "none_duration_s": a["duration_s"],
            "measured_duration_s": b["duration_s"],
            "none_min_clearance_m": a["min_clearance_m"],
            "measured_min_clearance_m": b["min_clearance_m"],
            "none_planning_mean_ms": a["mean_planning_ms"],
            "measured_planning_mean_ms": b["mean_planning_ms"],
            "measured_planning_p95_ms": b["p95_planning_ms"],
            "measured_planning_p99_ms": b["p99_planning_ms"],
            "measured_over100_fraction": b["over_100ms_fraction"],
            "measured_missed_control_ticks": b["missed_control_ticks"],
            "measured_submitted_commands": b["submitted_commands"],
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("results/phase17-paired.csv"))
    args = parser.parse_args()
    rows = []
    for root in args.roots:
        with (root / "episodes.csv").open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    paired = pair_rows(rows)
    if not paired:
        parser.error("no paired trials")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    for p in paired:
        print(f"{p['case_id']}/{p['mode']}: "
              f"none(success={p['none_success']}, collision={p['none_collision']}) "
              f"measured(success={p['measured_success']}, "
              f"collision={p['measured_collision']}, "
              f"missed={p['measured_missed_control_ticks']})")


if __name__ == "__main__":
    main()
