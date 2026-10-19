"""Attribute strict two-stage trajectory rejections to motion phases.

Diagnostic-only: no feasibility check, path, or candidate ranking is changed.
Counts are evaluated in the existing discrete CV and conservative-disc model.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .evaluation import TrialSpec, make_trial
from .experiment import RunConfig, run

PHASES = ("prefix", "first", "second", "brake", "hold")
CAUSES = ("road", "obstacle")


def rejection_windows(trace: list[dict], *, span_s: float = 5.0) -> list[dict]:
    if span_s <= 0:
        raise ValueError("window span must be positive")
    valid = [r for r in trace if isinstance(r.get("command_speed"), (int, float))]
    if not valid:
        return []
    start = float(valid[0]["time"])
    end = float(valid[-1]["time"])
    output = []
    k = 0
    while start + k * span_s <= end + 1e-9:
        from_s = start + k * span_s
        chunk = [r for r in valid if from_s <= float(r["time"]) < from_s + span_s]
        if chunk:
            row = {
                "from_s": from_s,
                "to_s": from_s + span_s,
                "n": len(chunk),
                "x_progress_m": round(float(chunk[-1]["x"]) - float(chunk[0]["x"]), 3),
                "mean_feasible_pairs": round(
                    sum(float(v.get("feasible_pairs", 0)) for v in chunk)/len(chunk), 3),
                "moving_first_available_fraction": round(
                    sum(bool(v.get("moving_first_action_feasible", False))
                        for v in chunk)/len(chunk), 3),
                "mean_target_speed_mps": round(
                    sum(float(v["command_speed"]) for v in chunk)/len(chunk), 3),
                "emergency_fraction": round(
                    sum(bool(v.get("emergency", False)) for v in chunk)/len(chunk), 3),
            }
            for phase in PHASES:
                for cause in CAUSES:
                    key = f"reject_{phase}_{cause}"
                    row[key] = sum(int(v.get(key, 0)) for v in chunk)
            output.append(row)
        k += 1
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Forensic rejection chronology")
    parser.add_argument("--scenario", default="two_targets")
    parser.add_argument("--stress", default="nominal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=30.0)
    parser.add_argument("--output", default="results/rejection-forensics")
    args = parser.parse_args(argv)
    trial = make_trial(TrialSpec(args.scenario, args.seed, args.stress))
    result = run(args.scenario, args.seed, "two_stage",
                 vehicle=trial.vehicle, planner=trial.planner,
                 config=RunConfig(max_time=args.max_time), world_override=trial.world)
    windows = rejection_windows(result.trace)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if windows:
        with (output/"rejection_windows.csv").open("w",newline="",encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(windows[0]))
            writer.writeheader()
            writer.writerows(windows)
    record={"scenario":args.scenario, "stress":args.stress,"seed":args.seed,
            "success":result.success,"collision":result.collision,
            "timeout":result.timeout,"road_exit":result.road_exit,
            "windows":windows}
    (output/"rejection_windows.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
