"""Closed-loop simulation, reproducible logging and metrics."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import json
import math
from pathlib import Path

from .model import Command, State, VehicleConfig
from .planner import Plan, PlannerConfig, choose_action
from .world import World, scenario
from .safety import CommandDelayLine
from .goal import goal_metrics


@dataclass(frozen=True)
class RunConfig:
    dt: float = 0.1
    max_time: float = 35.0
    start_speed: float = 1.5
    goal_tolerance: float = 0.6
    physical_road_footprint: str = "disc"
    goal_criterion: str = "x_gate"
    goal_lateral_tolerance_m: float = 0.75
    goal_heading_tolerance_deg: float = 15.0
    # None means drive-through section; set a limit for a docking experiment.
    goal_speed_limit_mps: float | None = None


@dataclass
class Result:
    scenario: str
    seed: int
    mode: str
    success: bool
    collision: bool
    road_exit: bool
    timeout: bool
    duration_s: float
    min_clearance_m: float
    emergency_steps: int
    stopped_steps: int
    mean_planning_ms: float
    backup_rejections: int
    unsafe_fallback_steps: int
    nominal_no_backup_steps: int
    control_delay_s: float
    trace: list[dict]
    decision_snapshot: dict | None = None
    reached_historical_x_line: bool = False
    reached_pose_gate: bool = False
    final_body_center_y_m: float = 0.0
    final_abs_heading_error_deg: float = 0.0
    goal_criterion: str = "x_gate"

    def metrics(self) -> dict:
        return {key: value for key, value in asdict(self).items() if key not in ("trace", "decision_snapshot")}


def run(scenario_name: str = "crossing", seed: int = 0, mode: str = "predictive",
        *, vehicle: VehicleConfig | None = None, planner: PlannerConfig | None = None,
        config: RunConfig | None = None,
        world_override: World | None = None) -> Result:
    vcfg = vehicle or VehicleConfig()
    pcfg = planner or PlannerConfig()
    rcfg = config or RunConfig()
    if rcfg.physical_road_footprint not in ("disc", "rectangle", "rearaxle_rectangle"):
        raise ValueError("unknown physical road footprint")
    if rcfg.goal_criterion not in ("x_gate", "pose_gate"):
        raise ValueError("unknown goal criterion")
    physical_vehicle = replace(vcfg, road_footprint=rcfg.physical_road_footprint)
    # Inject the same immutable world across paired planners; the simulator
    # never changes the world, so each mode receives identical target truth.
    world = world_override if world_override is not None else scenario(scenario_name, seed)
    state = State(0.0, 0.0, 0.0, rcfg.start_speed)
    min_clearance = float("inf")
    emergency_steps = stopped_steps = backup_rejections = unsafe_fallback_steps = nominal_no_backup_steps = 0
    actuator = CommandDelayLine(Command(rcfg.start_speed, 0.0), pcfg.control_delay_s)
    plan_time = 0.0
    trace = []
    decision_snapshot = None
    collision = road_exit = success = False
    reached_historical_x_line = reached_pose_gate = False
    final_goal = None
    steps = math.ceil(rcfg.max_time / rcfg.dt)
    for i in range(steps + 1):
        physical_road = world.road_clearance(state, physical_vehicle)
        physical_obstacle = world.obstacle_clearance(state, physical_vehicle)
        min_clearance = min(min_clearance, physical_road, physical_obstacle)
        collision = collision or physical_obstacle < 0
        road_exit = road_exit or physical_road < 0
        status = goal_metrics(
            state, world, physical_vehicle,
            distance_tolerance_m=rcfg.goal_tolerance,
            lateral_tolerance_m=rcfg.goal_lateral_tolerance_m,
            heading_tolerance_deg=rcfg.goal_heading_tolerance_deg,
            speed_limit_mps=rcfg.goal_speed_limit_mps)
        reached_historical_x_line |= status["historical_x_line_met"]
        reached_pose_gate |= status["pose_gate_met"]
        final_goal = status
        accepted_goal = (status["historical_x_line_met"]
                         if rcfg.goal_criterion == "x_gate"
                         else status["pose_gate_met"])
        # A physically collided/exited vehicle never counts as a success.
        success = accepted_goal and not collision and not road_exit
        if success or collision or road_exit or i == steps:
            trace.append({"time": state.time, "x": state.x, "y": state.y,
                          "yaw": state.yaw, "speed": state.speed, "steer": state.steer,
                          "min_clearance": min(physical_road, physical_obstacle),
                          "body_center_x": status["body_center_x_m"],
                          "body_center_y": status["body_center_y_m"],
                          "abs_heading_error_deg": status["abs_heading_error_deg"],
                          "historical_x_line_met": int(status["historical_x_line_met"]),
                          "pose_gate_met": int(status["pose_gate_met"]),
                          "command_speed": "", "command_steer": "", "applied_speed": "", "applied_steer": "", "pending_commands": len(actuator.pending), "backup_rejections": "", "unsafe_fallback": "", "fallback_strategy": "", "emergency": ""})
            break
        plan: Plan = choose_action(state, world, vcfg, pcfg, mode=mode, delay_line=actuator)
        if decision_snapshot is None and (
                any(d["status"] != "eligible" for d in plan.diagnostics)
                or state.x >= world.goal_x * .35):
            decision_snapshot = {
                "time": state.time, "x": state.x, "y": state.y,
                "yaw": state.yaw, "speed": state.speed, "mode": mode,
                "rows": plan.diagnostics,
            }
        feasible_first = [
            float(row["target_speed"]) for row in plan.diagnostics
            if row.get("status") in ("eligible", "selected")
            and row.get("continuation_count", 0) > 0
        ]
        max_feasible_first = max(feasible_first) if feasible_first else None
        emergency_steps += int(plan.emergency)
        backup_rejections += plan.rejected_by_backup
        unsafe_fallback_steps += int(plan.unsafe_fallback)
        nominal_no_backup_steps += int(plan.fallback_strategy == "nominal_no_backup")
        stopped_steps += int(state.speed < 0.1)
        plan_time += plan.compute_ms
        trace.append({"time": state.time, "x": state.x, "y": state.y,
                      "yaw": state.yaw, "speed": state.speed, "steer": state.steer,
                      "min_clearance": min(physical_road, physical_obstacle),
                      "body_center_x": status["body_center_x_m"],
                      "body_center_y": status["body_center_y_m"],
                      "abs_heading_error_deg": status["abs_heading_error_deg"],
                      "historical_x_line_met": int(status["historical_x_line_met"]),
                      "pose_gate_met": int(status["pose_gate_met"]),
                      "command_speed": plan.command.target_speed,
                      "command_steer": plan.command.target_steer,
                      "applied_speed": actuator.active.target_speed,
                      "applied_steer": actuator.active.target_steer,
                      "pending_commands": len(actuator.pending),
                      "backup_rejections": plan.rejected_by_backup,
                      "feasible_pairs": plan.feasible_count,
                      "feasible_first_actions": len(feasible_first),
                      "max_feasible_first_speed": max_feasible_first,
                      "moving_first_action_feasible": int(
                          any(v > 0.1 for v in feasible_first)),
                      **plan.rejection_counts,
                      "unsafe_fallback": int(plan.unsafe_fallback),
                      "fallback_strategy": plan.fallback_strategy,
                      "emergency": int(plan.emergency),
                      **plan.cost_parts})
        actuator.enqueue(state.time, plan.command)
        substeps = actuator.advance_path(state, state.time + rcfg.dt, vcfg,
                                         min(pcfg.safety_dt, rcfg.dt))
        for substep in substeps[1:]:
            physical_road = world.road_clearance(substep, physical_vehicle)
            physical_obstacle = world.obstacle_clearance(substep, physical_vehicle)
            min_clearance = min(min_clearance, physical_road, physical_obstacle)
            state = substep
            if physical_road < 0 or physical_obstacle < 0:
                # End at the first detected intermediate collision, not at next 100ms tick.
                break
        if physical_road < 0 or physical_obstacle < 0:
            continue
    return Result(scenario_name, seed, mode, success, collision, road_exit,
                  not (success or collision or road_exit), state.time, min_clearance,
                  emergency_steps, stopped_steps,
                  plan_time / max(1, len(trace) - 1),
                  backup_rejections, unsafe_fallback_steps, nominal_no_backup_steps,
                  pcfg.control_delay_s, trace, decision_snapshot,
                  reached_historical_x_line, reached_pose_gate,
                  final_goal["body_center_y_m"],
                  final_goal["abs_heading_error_deg"], rcfg.goal_criterion)


def save_result(result: Result, directory: str | Path) -> tuple[Path, Path]:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"{result.scenario}_{result.mode}_{result.seed}"
    metrics = output / f"{prefix}_metrics.json"
    trajectory = output / f"{prefix}_trace.csv"
    metrics.write_text(json.dumps(result.metrics(), ensure_ascii=False, indent=2), encoding="utf-8")
    with trajectory.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(dict.fromkeys(key for row in result.trace for key in row))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.trace)
    if result.decision_snapshot:
        snapshot = output / f"{prefix}_decision_candidates.csv"
        with snapshot.open("w", newline="", encoding="utf-8") as f:
            fields = list(dict.fromkeys(key for row in result.decision_snapshot["rows"] for key in row))
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result.decision_snapshot["rows"])
    return metrics, trajectory


def batch_compare(scenario_names: tuple[str, ...], seeds: int = 5,
                  modes: tuple[str, ...] = ("reactive", "predictive"),
                  *, vehicle: VehicleConfig | None = None,
                  planner: PlannerConfig | None = None,
                  config: RunConfig | None = None) -> list[dict]:
    if seeds < 1:
        raise ValueError("seeds must be positive")
    records = []
    for name in scenario_names:
        for seed in range(seeds):
            for mode in modes:
                records.append(run(name, seed, mode,
                                   vehicle=vehicle, planner=planner,
                                   config=config).metrics())
    return records
