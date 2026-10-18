"""Closed-loop simulation, reproducible logging and metrics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path

from .model import State, VehicleConfig, step
from .planner import Plan, PlannerConfig, choose_action
from .world import World, scenario


@dataclass(frozen=True)
class RunConfig:
    dt: float = 0.1
    max_time: float = 35.0
    start_speed: float = 1.5
    goal_tolerance: float = 0.6


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
    trace: list[dict]

    def metrics(self) -> dict:
        return {key: value for key, value in asdict(self).items() if key != "trace"}


def run(scenario_name: str = "crossing", seed: int = 0, mode: str = "predictive",
        *, vehicle: VehicleConfig | None = None, planner: PlannerConfig | None = None,
        config: RunConfig | None = None) -> Result:
    vcfg = vehicle or VehicleConfig()
    pcfg = planner or PlannerConfig()
    rcfg = config or RunConfig()
    world = scenario(scenario_name, seed)
    state = State(0.0, 0.0, 0.0, rcfg.start_speed)
    min_clearance = float("inf")
    emergency_steps = stopped_steps = 0
    plan_time = 0.0
    trace = []
    collision = road_exit = success = False
    steps = math.ceil(rcfg.max_time / rcfg.dt)
    for i in range(steps + 1):
        physical_road = world.road_clearance(state, vcfg)
        physical_obstacle = world.obstacle_clearance(state, vcfg)
        min_clearance = min(min_clearance, physical_road, physical_obstacle)
        collision = collision or physical_obstacle < 0
        road_exit = road_exit or physical_road < 0
        success = state.x >= world.goal_x - rcfg.goal_tolerance
        if success or collision or road_exit or i == steps:
            trace.append({"time": state.time, "x": state.x, "y": state.y,
                          "yaw": state.yaw, "speed": state.speed, "steer": state.steer,
                          "min_clearance": min(physical_road, physical_obstacle),
                          "command_speed": "", "command_steer": "", "emergency": ""})
            break
        plan: Plan = choose_action(state, world, vcfg, pcfg, mode=mode)
        emergency_steps += int(plan.emergency)
        stopped_steps += int(state.speed < 0.1)
        plan_time += plan.compute_ms
        trace.append({"time": state.time, "x": state.x, "y": state.y,
                      "yaw": state.yaw, "speed": state.speed, "steer": state.steer,
                      "min_clearance": min(physical_road, physical_obstacle),
                      "command_speed": plan.command.target_speed,
                      "command_steer": plan.command.target_steer,
                      "emergency": int(plan.emergency)})
        state = step(state, plan.command, vcfg, rcfg.dt)
    return Result(scenario_name, seed, mode, success, collision, road_exit,
                  not (success or collision or road_exit), state.time, min_clearance,
                  emergency_steps, stopped_steps,
                  plan_time / max(1, len(trace) - 1), trace)


def save_result(result: Result, directory: str | Path) -> tuple[Path, Path]:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"{result.scenario}_{result.mode}_{result.seed}"
    metrics = output / f"{prefix}_metrics.json"
    trajectory = output / f"{prefix}_trace.csv"
    metrics.write_text(json.dumps(result.metrics(), ensure_ascii=False, indent=2), encoding="utf-8")
    with trajectory.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.trace[0]))
        writer.writeheader()
        writer.writerows(result.trace)
    return metrics, trajectory


def batch_compare(scenario_names: tuple[str, ...], seeds: int = 5) -> list[dict]:
    if seeds < 1:
        raise ValueError("seeds must be positive")
    records = []
    for name in scenario_names:
        for seed in range(seeds):
            for mode in ("reactive", "predictive"):
                records.append(run(name, seed, mode).metrics())
    return records
