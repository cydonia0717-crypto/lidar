"""CLI demonstration and controlled baseline comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .experiment import RunConfig, batch_compare, run, save_result
from .model import Command, State, VehicleConfig
from .planner import PlannerConfig
from .safety import CommandDelayLine
from .world import scenario
from .visualize import save_figure, save_backup_figure, save_decision_figure


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Synthetic 2D Ackermann dynamic-obstacle avoidance simulator")
    parser.add_argument("--scenario", choices=("crossing", "two_targets", "empty", "narrow", "late_crossing"), default="crossing")
    parser.add_argument("--mode", choices=("reactive", "predictive", "delay_aware", "braking", "braking_hold", "two_stage", "two_stage_rectroad", "two_stage_bodyframe", "two_stage_viability"), default="predictive")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goal-criterion", choices=("x_gate", "pose_gate"), default="x_gate")
    parser.add_argument("--goal-lateral-tolerance", type=float, default=.75)
    parser.add_argument("--goal-heading-tolerance-deg", type=float, default=15.)
    parser.add_argument("--physical-road-footprint",
                        choices=("disc", "rectangle", "rearaxle_rectangle"), default="disc",
                        help="Physical collision ground truth; use rectangle for fair geometry ablations")
    parser.add_argument("--output", default="results")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--decision-figure", action="store_true",
                        help="Plot recorded candidate scoring and rejection decisions")
    parser.add_argument("--backup-figure", action="store_true",
                        help="Render the three motion phases for one hypothetical command")
    parser.add_argument("--control-delay", type=float, default=0.25)
    parser.add_argument("--deceleration", type=float, default=1.6)
    parser.add_argument("--execute-window", type=float, default=0.40)
    parser.add_argument("--second-window", type=float, default=1.0)
    parser.add_argument("--stationary-hold", type=float, default=6.0,
                        help="How long to check the stopping position after reaching rest")
    parser.add_argument("--clearance-weight", type=float, default=0.6)
    parser.add_argument("--heading-weight", type=float, default=0.4)
    parser.add_argument("--compare", action="store_true", help="Run paired scenarios with multiple planners")
    parser.add_argument("--compare-modes", default="reactive,predictive,delay_aware,braking,braking_hold,two_stage,two_stage_rectroad,two_stage_bodyframe,two_stage_viability",
                        help="Comma-separated planner modes for paired comparison")
    parser.add_argument("--seeds", type=int, default=5, help="Number of paired random seeds for --compare")
    args = parser.parse_args(argv)
    if args.control_delay < 0 or args.deceleration <= 0 or args.execute_window <= 0 or args.second_window <= 0 or args.stationary_hold < 0:
        parser.error("delay must be nonnegative, deceleration and execute-window must be positive")
    if args.clearance_weight < 0 or args.heading_weight < 0:
        parser.error("objective weights must be nonnegative")
    vehicle = VehicleConfig(braking=args.deceleration)
    planner = PlannerConfig(control_delay_s=args.control_delay,
                            candidate_execute_s=args.execute_window,
                            two_stage_second_s=args.second_window,
                            stationary_hold_s=args.stationary_hold,
                            clearance_weight=args.clearance_weight,
                            heading_weight=args.heading_weight)
    physical = RunConfig(physical_road_footprint=args.physical_road_footprint,
                         goal_criterion=args.goal_criterion,
                         goal_lateral_tolerance_m=args.goal_lateral_tolerance,
                         goal_heading_tolerance_deg=args.goal_heading_tolerance_deg)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.compare:
        modes = tuple(s.strip() for s in args.compare_modes.split(","))
        if any(mode not in ("reactive", "predictive", "delay_aware", "braking", "braking_hold", "two_stage", "two_stage_rectroad") for mode in modes):
            parser.error("invalid --compare-modes")
        records = batch_compare((args.scenario,), seeds=args.seeds, modes=modes,
                                vehicle=vehicle, planner=planner, config=physical)
        path = output / f"{args.scenario}_paired_comparison.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"Synthetic paired results: {path}")
        for mode in modes:
            subset = [r for r in records if r["mode"] == mode]
            n = len(subset)
            successes = [r["duration_s"] for r in subset if r["success"]]
            mean_success = sum(successes) / len(successes) if successes else float("nan")
            print(f"{mode}: success={len(successes)}/{n}, "
                  f"collision={sum(r['collision'] for r in subset)}/{n}, "
                  f"successful_mean_duration={mean_success:.2f}s, "
                  f"backup_rejections={sum(r['backup_rejections'] for r in subset)}")
        return
    result = run(args.scenario, args.seed, args.mode,
                 vehicle=vehicle, planner=planner, config=physical)
    paths = save_result(result, output)
    print(json.dumps(result.metrics(), indent=2))
    print(f"Wrote metrics and trace: {paths[0]}, {paths[1]}")
    if args.plot:
        fig = save_figure(result, output / f"{args.scenario}_{args.mode}_{args.seed}.png")
        print(f"Wrote figure: {fig}")
    if args.decision_figure:
        if result.decision_snapshot:
            fig = save_decision_figure(result,
                    output / f"{args.scenario}_{args.mode}_{args.seed}_decision.png")
            print(f"Wrote decision explanation: {fig}")
        else:
            print("No decision snapshot could be recorded in this run.")
    if args.backup_figure:
        initial = State(0.0, 0.0, 0.0, 1.5)
        line = CommandDelayLine(Command(initial.speed, initial.steer), planner.control_delay_s)
        fig = save_backup_figure(initial, Command(vehicle.max_speed, 0.0), line,
                                 scenario(args.scenario, args.seed), vehicle, planner,
                                 output / f"{args.scenario}_three_stage_backup.png")
        print(f"Wrote three-stage preview (not executed): {fig}")


if __name__ == "__main__":
    main()
