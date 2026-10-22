"""Audit COMPLETE closed-loop equivalence and offline runtime deadline misses.

Two modes have identical hard safety filters and candidate ranking. This
script intentionally fails when any commanded or physical state differs. It
uses per-step planner wall-time, NOT a model of onboard compute latency.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

STATE_COMMAND_KEYS = (
    "time", "x", "y", "yaw", "speed", "steer",
    "body_center_x", "body_center_y", "abs_heading_error_deg",
    "command_speed", "command_steer", "applied_speed", "applied_steer",
    "pending_commands", "unsafe_fallback", "fallback_strategy", "emergency",
)


def percentile(values: list[float], fraction: float) -> float:
    if not values or not 0 <= fraction <= 1:
        raise ValueError("nonempty times and fraction in [0,1] required")
    sorted_values=sorted(values)
    return sorted_values[max(0, math.ceil(fraction*len(sorted_values))-1)]


def read_rows(path: Path) -> list[dict[str,str]]:
    with path.open(newline="",encoding="utf8") as fh:
        return list(csv.DictReader(fh))


def paired_audit(roots: tuple[Path,...], *, deadline_ms: float=100.) -> list[dict]:
    if not roots or not math.isfinite(deadline_ms) or deadline_ms <= 0:
        raise ValueError("invalid input roots or control deadline")
    reports=[]
    for root in roots:
        traces=root/"all_traces"
        if not traces.is_dir():
            raise ValueError("missing full trajectories: "+str(traces))
        originals=sorted(traces.glob("*__two_stage_viability.csv"))
        if not originals:
            raise ValueError("missing phase15 full traces: "+str(traces))
        for slow_file in originals:
            fast_file=slow_file.with_name(
                slow_file.name.replace("__two_stage_viability.csv",
                                       "__two_stage_viability_fast.csv"))
            if not fast_file.is_file():
                raise ValueError("missing fast paired trace: "+str(fast_file))
            slow=read_rows(slow_file)
            fast=read_rows(fast_file)
            if len(slow) != len(fast):
                raise AssertionError(
                    f"divergent {slow_file.stem}: {len(slow)} vs {len(fast)} frames")
            for frame,(a,b) in enumerate(zip(slow,fast)):
                for name in STATE_COMMAND_KEYS:
                    if a.get(name)!=b.get(name):
                        raise AssertionError(
                            f"{slow_file.stem} frame {frame} changed {name}: "
                            f"{a.get(name)!r} != {b.get(name)!r}")
            report={"case_id":slow_file.stem.replace("__two_stage_viability",""),
                    "control_frames":len(slow)-1,
                    "all_commands_states_exactly_equal":True}
            for mode,rows in (("original",slow),("fast",fast)):
                times=[float(r["planning_compute_ms"]) for r in rows
                       if r.get("planning_compute_ms") not in ("",None)]
                if not times:
                    raise ValueError("missing per-control-step compute telemetry")
                report[mode+"_mean_ms"]=mean(times)
                report[mode+"_p95_ms"]=percentile(times,.95)
                report[mode+"_p99_ms"]=percentile(times,.99)
                report[mode+"_max_ms"]=max(times)
                report[mode+"_deadline_misses"]=sum(x>deadline_ms for x in times)
                report[mode+"_deadline_miss_fraction"]=(
                    report[mode+"_deadline_misses"]/len(times))
            report["paired_mean_speedup"]=(
                report["original_mean_ms"]/report["fast_mean_ms"])
            reports.append(report)
    return reports


def main(argv=None) -> None:
    parser=argparse.ArgumentParser(
        description="Strict mode-equivalence and offline per-control-step CPU budget audit")
    parser.add_argument("--roots",
                        default="results/phase16-known,results/phase16-holdout,results/phase16-other")
    parser.add_argument("--deadline-ms",type=float,default=100.)
    parser.add_argument("--output",default="results/phase16-audit")
    args=parser.parse_args(argv)
    reports=paired_audit(
        tuple(Path(p.strip()) for p in args.roots.split(",") if p.strip()),
        deadline_ms=args.deadline_ms)
    folder=Path(args.output)
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/"paired_exact_trajectories_and_runtime.csv").open(
            "w",newline="",encoding="utf8") as fh:
        wr=csv.DictWriter(fh,fieldnames=list(reports[0]))
        wr.writeheader()
        wr.writerows(reports)
    (folder/"audit_summary.json").write_text(json.dumps({
        "source":"GitHub Runner wall-clock time, NOT onboard runtime",
        "deadline_ms":args.deadline_ms,
        "fully_equal_pairs":len(reports),
        "all_closed_loop_states_and_commands_equal":True,
        "paired_cases":reports,
        "warnings": [
          "Simulated physical control execution is fixed-rate; elapsed CPU compute times "
          "do not delay the command queue.",
          "This audit is not a formal safety proof and does not establish Jetson deadlines."
        ]},indent=2,ensure_ascii=False),encoding="utf8")
    for row in reports:
        print(f"{row['case_id']}: exact=True "
              f"slow={row['original_mean_ms']:.2f}ms "
              f"fast={row['fast_mean_ms']:.2f}ms "
              f"speedup={row['paired_mean_speedup']:.2f}x "
              f"fast_p95={row['fast_p95_ms']:.2f}ms "
              f"deadline_miss={row['fast_deadline_misses']}/{row['control_frames']}",
              flush=True)


if __name__=="__main__":
    main()
