"""Explain which part of the TWO-STAGE closed loop stalls, using raw trace.

No conclusions are inserted by hand: all numbers originate from executed
synthetic trajectories. An often-planned second action is not an executed
command; the simulation only ever sends the first stage.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from .evaluation import TrialSpec, make_trial
from .experiment import RunConfig, run


def stall_windows(trace: list[dict], *, window_s: float = 5.0) -> list[dict]:
    if not trace:
        raise ValueError("trace is empty")
    if window_s <= 0:
        raise ValueError("window must be positive")
    valid = [r for r in trace if isinstance(r.get("command_speed"), (float, int))]
    if not valid:
        return []
    results = []
    t0 = float(valid[0]["time"])
    tmax = float(valid[-1]["time"])
    k = 0
    while t0 + k * window_s <= tmax + 1e-9:
        left = t0 + k * window_s
        chunk = [r for r in valid if left <= float(r["time"]) < left + window_s]
        if chunk:
            zero_cmd = sum(float(r["command_speed"]) < .1 for r in chunk)
            emergency = sum(bool(r.get("emergency", False)) for r in chunk)
            at_rest = sum(float(r["speed"]) < .1 for r in chunk)
            second = [float(r["planned_second_speed"]) for r in chunk
                      if isinstance(r.get("planned_second_speed"), (int, float))]
            first_progress = [float(r["planned_first_progress_m"]) for r in chunk
                              if isinstance(r.get("planned_first_progress_m"), (int, float))]
            second_progress = [float(r["planned_second_progress_m"]) for r in chunk
                               if isinstance(r.get("planned_second_progress_m"), (int, float))]
            results.append({
                "from_s": round(left, 3),
                "to_s": round(left + window_s, 3),
                "n": len(chunk),
                "x_displacement_m": round(float(chunk[-1]["x"]) - float(chunk[0]["x"]), 4),
                "mean_actual_speed_mps": round(mean(float(r["speed"]) for r in chunk), 4),
                "zero_first_command_fraction": round(zero_cmd / len(chunk), 3),
                "rest_fraction": round(at_rest / len(chunk), 3),
                "emergency_fraction": round(emergency / len(chunk), 3),
                "planned_second_mean_speed_mps": round(mean(second), 4) if second else None,
                "planned_first_progress_mean_m": round(mean(first_progress), 4) if first_progress else None,
                "planned_second_progress_mean_m": round(mean(second_progress), 4) if second_progress else None,
            })
        k += 1
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit stagnant two-stage rollouts")
    parser.add_argument("--scenario", default="two_targets")
    parser.add_argument("--stress", default="nominal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("two_stage", "two_stage_commit"), default="two_stage")
    parser.add_argument("--max-time", type=float, default=30.0)
    parser.add_argument("--output", default="results/stall-analysis")
    args = parser.parse_args(argv)
    trial = make_trial(TrialSpec(args.scenario, args.seed, args.stress))
    res = run(args.scenario, args.seed, args.mode, vehicle=trial.vehicle,
              planner=trial.planner, config=RunConfig(max_time=args.max_time),
              world_override=trial.world)
    windows = stall_windows(res.trace)
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    if windows:
        with (folder / "stall_windows.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(windows[0]))
            writer.writeheader()
            writer.writerows(windows)
    payload = {"scenario": args.scenario, "stress": args.stress,
               "seed": args.seed, "mode": args.mode,
               "success": res.success, "collision": res.collision,
               "timeout": res.timeout, "road_exit": res.road_exit,
               "duration_s": res.duration_s, "windows": windows}
    (folder / "stall_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
