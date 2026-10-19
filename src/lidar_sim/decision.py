"""Explainable candidate trajectory objective for the simulation controller.

Hard safety constraints are evaluated by planner.py independently of this score.
All output components are already weighted costs; lower is better.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .model import Command, State, VehicleConfig
from .world import World


@dataclass(frozen=True)
class CostBreakdown:
    route: float
    progress: float
    speed_change: float
    steering_change: float
    heading: float
    clearance: float

    @property
    def total(self) -> float:
        return sum(asdict(self).values())

    def as_record(self) -> dict[str, float]:
        parts = asdict(self)
        return {"cost_" + key: value for key, value in parts.items()} | {"total_cost": self.total}


def trajectory_cost_breakdown(path: list[State], command: Command, start: State,
                              world: World, vehicle: VehicleConfig, cfg,
                              *, predict: bool = True, padding: float = 0.0) -> CostBreakdown:
    if not path:
        raise ValueError("trajectory must not be empty")
    if cfg.clearance_target_m <= 0 or cfg.clearance_weight < 0 or cfg.heading_weight < 0:
        raise ValueError("clearance target and objective weights must be valid")

    route = cfg.route_weight * sum(s.y * s.y for s in path) / len(path)
    progress = -cfg.progress_weight * max(0.0, min(world.goal_x, path[-1].x) - start.x)
    speed_change = cfg.speed_change_weight * (command.target_speed - start.speed) ** 2
    steering_change = cfg.steer_weight * (command.target_steer - start.steer) ** 2
    heading = cfg.heading_weight * sum(s.yaw ** 2 for s in path) / len(path)

    # Penalize trajectories close to dynamic targets *or* the road boundary,
    # while preserving the separately enforced hard collision constraints.
    # The per-sample cost uses the same prediction time basis as the hard check.
    proximity = 0.0
    fixed_time = path[0].time
    for s in path:
        free = world.road_clearance(s, vehicle) - world.safety_margin
        for obs in world.obstacles:
            ox, oy = obs.position(s.time if predict else fixed_time)
            dist = (math.hypot(s.x - ox, s.y - oy) - vehicle.collision_radius -
                    obs.radius - world.safety_margin - padding)
            free = min(free, dist)
        proximity += max(0.0, cfg.clearance_target_m - free) ** 2
    clearance = cfg.clearance_weight * proximity / len(path)
    return CostBreakdown(route, progress, speed_change, steering_change,
                         heading, clearance)
