"""Per-episode phase16 workload decomposition from logged decision traces.

Source is the same GitHub Runner/host as the plan wall-clock measurement.
Diagnostic clocks can overlap by instrumentation overhead; fractions are
descriptive, not an exact profiler or a Jetson runtime guarantee.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


def profile(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as f:
        decisions = [r for r in csv.DictReader(f)
                     if r.get("planning_ms") not in (None, "")]
    if not decisions or "lookahead_parent_ms" not in decisions[0]:
        raise ValueError(f"{path}: missing phase16 per-decision diagnostics")
    def values(key: str) -> list[float]:
        return [float(r.get(key) or 0) for r in decisions]
    total_ms = sum(values("planning_ms"))
    parent_ms = sum(values("lookahead_parent_ms"))
    child_ms = sum(values("lookahead_child_ms"))
    n = len(decisions)
    return {
        "trace": str(path),
        "decisions": n,
        "mean_planning_ms": total_ms / n,
        "mean_parent_ms": parent_ms / n,
        "mean_child_ms": child_ms / n,
        "child_fraction_of_total": child_ms / total_ms if total_ms else 0.0,
        "child_calls": int(sum(values("lookahead_child_calls"))),
        "candidate_probes": int(sum(values("lookahead_probed"))),
        "skipped_candidate_probes": int(sum(values("lookahead_unprobed"))),
        "early_exit_ticks": int(sum(values("lookahead_early_exit"))),
        "cached_alternative_ticks": int(sum(values("lookahead_cached_alternative"))),
        "no_viable_shortlist_ticks": int(sum(values("lookahead_no_viable_shortlist"))),
        "mean_cache_entries": mean(values("lookahead_cache_entries")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("results/phase16-profile.csv"))
    args = parser.parse_args()
    files = sorted({p for root in args.roots
                    for p in (root.rglob("all_traces/*__two_stage_viability.csv")
                              if root.is_dir() else [root])})
    if not files:
        parser.error("no optimized phase16 full traces found")
    rows = [profile(p) for p in files]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['trace']}: parent={row['mean_parent_ms']:.1f}ms "
              f"child={row['mean_child_ms']:.1f}ms "
              f"child_calls={row['child_calls']} "
              f"skipped={row['skipped_candidate_probes']} "
              f"cache_reuse={row['cached_alternative_ticks']}")


if __name__ == "__main__":
    main()
