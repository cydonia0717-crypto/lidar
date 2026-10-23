"""Phase17 deterministic fixed-lag sweep audit.

Never collapse this deliberately non-monotone phase space into a single safe
latency threshold. Same nominal fixed lag is imposed on each planner mode.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .phase17_forensics import final_parked_since, first_event, load


def analyze_sweep(root: Path) -> list[dict]:
    with (root / "episodes.csv").open(newline="", encoding="utf-8") as f:
        episodes = list(csv.DictReader(f))
    if not episodes:
        raise ValueError("no fixed-lag episodes")
    grouped: dict[tuple[str, str], list[dict]] = {}
    for e in episodes:
        if e["compute_policy"] != "fixed":
            raise ValueError("fixed-lag audit received non-fixed policy")
        key = (e["case_id"], e["mode"])
        grouped.setdefault(key, []).append(e)
    lag_sets = []
    reports = []
    for (case_id, mode), records in sorted(grouped.items()):
        rows = sorted(records, key=lambda r: float(r["fixed_compute_ms"]))
        lag_values = [float(r["fixed_compute_ms"]) for r in rows]
        if any(not math.isfinite(v) or v < 0 for v in lag_values):
            raise ValueError("invalid fixed lag")
        if len(set(lag_values)) != len(lag_values):
            raise ValueError(f"duplicated lag: {case_id}/{mode}")
        lag_sets.append((case_id, lag_values))
        prior = None
        for record in rows:
            lag = float(record["fixed_compute_ms"])
            subdir = root / case_id / mode / f"fixed_{lag:g}ms"
            scenario, _, seed = case_id.rpartition("__")
            # Filename uses the scenario label, not scenario__stress.
            scenario_name = case_id.split("__")[0]
            path = subdir / f"{scenario_name}_{mode}_{seed}_trace.csv"
            metric_path = path.with_name(path.name.replace(
                "_trace.csv", "_metrics.json"))
            trace = load(path)
            if not trace:
                raise ValueError(f"empty trace: {path}")
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
            for field in ("success", "collision", "timeout", "road_exit"):
                if str(metrics[field]).lower() != str(record[field]).lower():
                    raise AssertionError(f"{case_id}/{mode}/{lag}: {field} inconsistent")
            issued = [r for r in trace if r.get("compute_busy") == "0"]
            if len(issued) != int(record["submitted_commands"]):
                raise AssertionError(f"{path}: command count inconsistent")
            for row in issued:
                if abs(float(row["compute_submit_delay_ms"]) - lag) > 1e-8:
                    raise AssertionError(f"{path}: injected compute lag mismatch")
            parked = final_parked_since(trace)
            final_time = float(trace[-1]["time"])
            collision = str(record["collision"]).lower() == "true"
            result = {
                "case_id": case_id, "mode": mode,
                "fixed_compute_ms": lag,
                "success": str(record["success"]).lower() == "true",
                "collision": collision,
                "timeout": str(record["timeout"]).lower() == "true",
                "road_exit": str(record["road_exit"]).lower() == "true",
                "terminal_time_s": final_time,
                "min_clearance_m": float(record["min_clearance_m"]),
                "missed_control_ticks": int(record["missed_control_ticks"]),
                "issued_commands": len(issued),
                "first_unsafe_fallback_s": first_event(trace, "unsafe_fallback"),
                "parked_since_s": parked,
                "stationary_contact_after_2s": bool(
                    collision and parked is not None
                    and final_time - parked >= 2.0),
                "changed_outcome_from_previous_lag": (
                    False if prior is None
                    else (str(prior["success"]).lower() != str(record["success"]).lower()
                          or str(prior["collision"]).lower() != str(record["collision"]).lower()
                          or str(prior["timeout"]).lower() != str(record["timeout"]).lower())
                ),
            }
            reports.append(result)
            prior = record
    cases = {case for case, _ in lag_sets}
    if len(cases) != 1:
        raise ValueError("fixed sweep must use one shared physical case")
    if len({tuple(lags) for _, lags in lag_sets}) != 1:
        raise ValueError("all planner modes must use identical fixed lag values")
    return reports


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=Path("results/phase17-fixed-sweep-audit.csv"))
    args = p.parse_args()
    rows = analyze_sweep(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for mode in sorted({r["mode"] for r in rows}):
        seq = [r for r in rows if r["mode"] == mode]
        print(f"{mode}: " + ", ".join(
            f"{r['fixed_compute_ms']:g}ms="
            + ("success" if r["success"] else
               "collision" if r["collision"] else
               "timeout" if r["timeout"] else "road-exit")
            for r in seq))
        print(f"  transitions={sum(r['changed_outcome_from_previous_lag'] for r in seq)}")
    print("No monotone safe-lag threshold is assumed or fitted.")


if __name__ == "__main__":
    main()
