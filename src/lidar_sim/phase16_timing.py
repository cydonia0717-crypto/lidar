"""Summarize measured planner wall time from full decision traces.

The final terminal-state row has no planning_ms and must not enter timing
statistics. These timings are host wall time, not plant control delays.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean


def percentile(values: list[float], q: float) -> float:
    if not values or not 0 <= q <= 100:
        raise ValueError("nonempty measurements and percentile in [0,100] required")
    a = sorted(values)
    k = (len(a) - 1) * q / 100
    lo = math.floor(k)
    hi = math.ceil(k)
    return a[lo] + (a[hi] - a[lo]) * (k - lo)


def summarize(path: Path, period_ms: float = 100.0) -> dict:
    if not math.isfinite(period_ms) or period_ms <= 0:
        raise ValueError("period_ms must be finite and positive")
    with path.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    if not records or "planning_ms" not in records[0]:
        raise ValueError(f"{path}: missing per-decision planning_ms; rerun with phase16 code")
    samples = []
    for row in records:
        raw = row.get("planning_ms", "")
        if raw == "":
            continue  # terminal state, not a decision
        ms = float(raw)
        if not math.isfinite(ms) or ms < 0:
            raise ValueError(f"{path}: invalid planning_ms={raw!r}")
        samples.append(ms)
    if not samples:
        raise ValueError(f"{path}: no planning samples")
    return {
        "trace": str(path),
        "decisions": len(samples),
        "mean_ms": mean(samples),
        "p95_ms": percentile(samples, 95),
        "p99_ms": percentile(samples, 99),
        "max_ms": max(samples),
        "over_period_count": sum(x > period_ms for x in samples),
        "over_period_fraction": sum(x > period_ms for x in samples) / len(samples),
        "period_ms": period_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--period-ms", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=Path("results/phase16-timing.csv"))
    args = parser.parse_args()
    paths = sorted({p for root in args.roots for p in
                    (root.rglob("all_traces/*.csv") if root.is_dir() else [root])})
    if not paths:
        parser.error("no trace files found")
    rows = [summarize(p, args.period_ms) for p in paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['trace']}: mean={row['mean_ms']:.2f}ms "
              f"p95={row['p95_ms']:.2f}ms p99={row['p99_ms']:.2f}ms "
              f"over-period={row['over_period_fraction']:.1%} "
              f"n={row['decisions']}")


if __name__ == "__main__":
    main()
