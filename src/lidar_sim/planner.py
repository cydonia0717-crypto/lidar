"""Finite-set candidate planning with time-aligned dynamic occupancy checks."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

from .model import Command, State, VehicleConfig, step
from .world import World


@dataclass(frozen=True)
class PlannerConfig:
    horizon: float = 3.0
    sample_dt: float = 0.15
    speed_samples: tuple[float, ...] = (0.0, 0.6, 1.2, 1.8, 2.4)
    steer_samples: tuple[float, ...] = (-0.42, -0.21, 0.0, 0.21, 0.42)
    route_weight: float = 1.4
    progress_weight: float = 3.5
    speed_change_weight: float = 0.15
    steer_weight: float = 0.2
    prediction_padding: float = 0.12


@dataclass
class Plan:
    command: Command
    trajectory: list[State]
    score: float
    feasible_count: int
    candidate_count: int
    compute_ms: float
    emergency: bool = False


def collision_free(trajectory: list[State], world: World, vehicle: VehicleConfig,
                   *, predict: bool = True, padding: float = 0.0) -> bool:
    """Check conservative disc-vs-disc/road collision on rollout samples.

    Padding is a modeling assumption, NOT empirically calibrated coverage.
    """
    for s in trajectory:
        if world.road_clearance(s, vehicle) < world.safety_margin:
            return False
        for obs in world.obstacles:
            ox, oy = obs.position(s.time if predict else trajectory[0].time)
            dist = math.hypot(s.x - ox, s.y - oy)
            if dist < vehicle.collision_radius + obs.radius + world.safety_margin + padding:
                return False
    return True


def rollout(initial: State, command: Command, vehicle: VehicleConfig,
            horizon: float, dt: float) -> list[State]:
    if horizon <= 0 or dt <= 0:
        raise ValueError("horizon and dt must be positive")
    path = [initial]
    while path[-1].time < initial.time + horizon - 1e-9:
        next_dt = min(dt, initial.time + horizon - path[-1].time)
        path.append(step(path[-1], command, vehicle, next_dt))
    return path


def trajectory_cost(path: list[State], command: Command, start: State,
                    world: World, cfg: PlannerConfig) -> float:
    avg_lateral = sum(p.y ** 2 for p in path) / len(path)
    forward_progress = max(0.0, min(path[-1].x, world.goal_x) - start.x)
    delta_speed = abs(command.target_speed - start.speed)
    delta_steer = abs(command.target_steer - start.steer)
    # Lower is better. Progress is a reward; comfort penalties discourage zigzagging.
    return (cfg.route_weight * avg_lateral - cfg.progress_weight * forward_progress
            + cfg.speed_change_weight * delta_speed ** 2
            + cfg.steer_weight * delta_steer ** 2)


def choose_action(initial: State, world: World, vehicle: VehicleConfig,
                  cfg: PlannerConfig, *, mode: str = "predictive") -> Plan:
    if mode not in ("reactive", "predictive"):
        raise ValueError(f"unknown planner mode: {mode}")
    start_time = time.perf_counter()
    best: tuple[float, Command, list[State]] | None = None
    count = 0
    total = 0
    for target_speed in cfg.speed_samples:
        for target_steer in cfg.steer_samples:
            total += 1
            command = Command(target_speed, target_steer)
            path = rollout(initial, command, vehicle, cfg.horizon, cfg.sample_dt)
            padding = cfg.prediction_padding if mode == "predictive" else 0.0
            if not collision_free(path, world, vehicle, predict=(mode == "predictive"), padding=padding):
                continue
            count += 1
            score = trajectory_cost(path, command, initial, world, cfg)
            if best is None or score < best[0]:
                best = (score, command, path)
    elapsed = (time.perf_counter() - start_time) * 1000
    if best is None:
        # A fail-safe *request*, not a guarantee: an unavoidable collision remains possible.
        emergency = Command(0.0, 0.0)
        path = rollout(initial, emergency, vehicle, cfg.horizon, cfg.sample_dt)
        return Plan(emergency, path, math.inf, 0, total, elapsed, emergency=True)
    return Plan(best[1], best[2], best[0], count, total, elapsed)
