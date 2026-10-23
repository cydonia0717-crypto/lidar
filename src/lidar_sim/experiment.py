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
from .simulated_perception import SimulatedPerception


@dataclass(frozen=True)
class RunConfig:
    dt: float = 0.1
    max_time: float = 35.0
    start_speed: float = 1.5
    goal_tolerance: float = 0.6
    physical_road_footprint: str = "disc"
    # None preserves historical coupling; otherwise physical truth is independent.
    physical_body_offset_m: float | None = None
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
    physical_body_offset_m: float = 0.55
    assumed_body_offset_m: float = 0.55
    # Optional phase18 observation-path telemetry (oracle runs are unchanged).
    perception_enabled: bool = False
    perception_blocked_steps: int = 0
    perception_planned_steps: int = 0
    perception_detection_events: int = 0

    def metrics(self) -> dict:
        return {key: value for key, value in asdict(self).items() if key not in ("trace", "decision_snapshot")}


def run(scenario_name: str = "crossing", seed: int = 0, mode: str = "predictive",
        *, vehicle: VehicleConfig | None = None, planner: PlannerConfig | None = None,
        config: RunConfig | None = None,
        world_override: World | None = None,
        perception: SimulatedPerception | None = None) -> Result:
    vcfg = vehicle or VehicleConfig()
    pcfg = planner or PlannerConfig()
    rcfg = config or RunConfig()
    if rcfg.physical_road_footprint not in ("disc", "rectangle", "rearaxle_rectangle"):
        raise ValueError("unknown physical road footprint")
    if rcfg.goal_criterion not in ("x_gate", "pose_gate"):
        raise ValueError("unknown goal criterion")
    if not math.isfinite(vcfg.rear_axle_to_body_center_m) or not 0 <= vcfg.rear_axle_to_body_center_m <= vcfg.length / 2:
        raise ValueError("invalid planning body offset")
    physical_offset = (vcfg.rear_axle_to_body_center_m
                       if rcfg.physical_body_offset_m is None
                       else rcfg.physical_body_offset_m)
    if not math.isfinite(physical_offset) or not 0 <= physical_offset <= vcfg.length / 2:
        raise ValueError("invalid physical body offset")
    physical_vehicle = replace(
        vcfg, road_footprint=rcfg.physical_road_footprint,
        rear_axle_to_body_center_m=physical_offset)
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
    ego_history: list[State] = []
    perception_blocked_steps = perception_planned_steps = perception_detection_events = 0
    for i in range(steps + 1):
        if perception is not None:
            # Historical self poses are revealed to the sensor at capture
            # time; future ego poses are never available to this provider.
            ego_history.append(state)
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
        planner_world = world
        perception_fields: dict = {}
        if perception is not None:
            observed = perception.observe(world, ego_history, dt=rcfg.dt)
            perception_detection_events += observed.observation_count
            perception_fields = {
                "perception_observed_at_s": (
                    observed.last_capture_time_s
                    if observed.last_capture_time_s is not None else ""),
                "perception_detection_events": observed.observation_count,
                "perception_valid_tracks": observed.valid_tracks,
                "perception_withheld_tracks": observed.withheld_tracks,
                "perception_stale_track_ids": "|".join(observed.stale_track_ids),
                "perception_max_track_age_s": observed.max_valid_age_s,
                "perception_blocked_reason": observed.blocked_reason or "",
                # No true target position, velocity or future trajectory
                # ever appears in the command-producing planner inputs.
                "perception_source": "causal_estimated_tracks",
            }
            if observed.blocked_reason:
                perception_blocked_steps += 1
                # Fails closed at the planner interface. Braking is bounded
                # and can still physically collide; never call it a guarantee.
                plan = Plan(
                    Command(0.0, state.steer), [], math.inf, 0, 0, 0.0,
                    emergency=True, unsafe_fallback=True,
                    fallback_strategy="perception_untrusted_brake",
                )
            else:
                planner_world = observed.estimated_world
                perception_planned_steps += 1
                plan = choose_action(
                    state, planner_world, vcfg, pcfg, mode=mode,
                    delay_line=actuator)
        else:
            # Unmodified oracle reference path for older phase1-17 results.
            plan = choose_action(state, world, vcfg, pcfg, mode=mode, delay_line=actuator)
        if perception is not None:
            # Evaluation ONLY, deliberately after the command-producing
            # planner call. These ground-truth errors must never feed back
            # into planning decisions or rejection thresholds.
            truth_by_name = {target.name: target for target in world.obstacles}
            position_errors = []
            velocity_errors = []
            for target in observed.estimated_world.obstacles:
                reference = truth_by_name[target.name]
                x_ref, y_ref = reference.position(state.time)
                x_est, y_est = target.position(state.time)
                position_errors.append(math.hypot(x_est-x_ref, y_est-y_ref))
                velocity_errors.append(math.hypot(
                    target.vx-reference.vx, target.vy-reference.vy))
            perception_fields["perception_eval_mean_position_error_m"] = (
                sum(position_errors)/len(position_errors) if position_errors else "")
            perception_fields["perception_eval_mean_velocity_error_mps"] = (
                sum(velocity_errors)/len(velocity_errors) if velocity_errors else "")
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
                      "planning_ms": plan.compute_ms,
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
                      **plan.cost_parts,
                      **perception_fields})
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
                  final_goal["abs_heading_error_deg"], rcfg.goal_criterion,
                  physical_offset, vcfg.rear_axle_to_body_center_m,
                  perception is not None, perception_blocked_steps,
                  perception_planned_steps, perception_detection_events)


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
