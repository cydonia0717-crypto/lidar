"""CLI demonstration and controlled baseline comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .experiment import batch_compare, run, save_result
from .visualize import save_figure


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Synthetic 2D Ackermann dynamic-obstacle avoidance simulator")
    parser.add_argument("--scenario", choices=("crossing", "two_targets", "empty", "narrow"), default="crossing")
    parser.add_argument("--mode", choices=("reactive", "predictive"), default="predictive")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Run identical scenarios and seeds with both planner modes")
    parser.add_argument("--seeds", type=int, default=5, help="Number of paired random seeds for --compare")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.compare:
        records = batch_compare((args.scenario,), seeds=args.seeds)
        path = output / f"{args.scenario}_paired_comparison.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"Synthetic paired results: {path}")
        for mode in ("reactive", "predictive"):
            subset = [r for r in records if r["mode"] == mode]
            n = len(subset)
            print(f"{mode}: success={sum(r['success'] for r in subset)}/{n}, "
                  f"collision={sum(r['collision'] for r in subset)}/{n}, "
                  f"mean_duration={sum(r['duration_s'] for r in subset)/n:.2f}s")
        return
    result = run(args.scenario, args.seed, args.mode)
    paths = save_result(result, output)
    print(json.dumps(result.metrics(), indent=2))
    print(f"Wrote metrics and trace: {paths[0]}, {paths[1]}")
    if args.plot:
        fig = save_figure(result, output / f"{args.scenario}_{args.mode}_{args.seed}.png")
        print(f"Wrote figure: {fig}")


if __name__ == "__main__":
    main()
