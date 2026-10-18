"""Deterministic synthetic crossing targets and road boundaries."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .model import State, VehicleConfig


@dataclass(frozen=True)
class MovingObstacle:
    name: str
    x0: float
    y0: float
    vx: float
    vy: float
    radius: float = 0.35

    def position(self, time: float) -> tuple[float, float]:
        return self.x0 + self.vx * time, self.y0 + self.vy * time


@dataclass(frozen=True)
class World:
    obstacles: tuple[MovingObstacle, ...]
    road_half_width: float = 3.25
    goal_x: float = 28.0
    safety_margin: float = 0.2

    def road_clearance(self, state: State, car: VehicleConfig) -> float:
        return self.road_half_width - abs(state.y) - car.collision_radius

    def obstacle_clearance(self, state: State, car: VehicleConfig, *, time: float | None = None) -> float:
        t = state.time if time is None else time
        if not self.obstacles:
            return float("inf")
        return min(math.hypot(state.x - obs.position(t)[0], state.y - obs.position(t)[1])
                   - car.collision_radius - obs.radius for obs in self.obstacles)

    def physical_clearance(self, state: State, car: VehicleConfig) -> float:
        return min(self.road_clearance(state, car), self.obstacle_clearance(state, car))


def scenario(name: str = "crossing", seed: int = 0) -> World:
    rng = random.Random(seed)
    jitter = rng.uniform(-0.7, 0.7)
    if name == "crossing":
        # Ego traveling toward +x; pedestrian crosses at x≈13 around t≈8 s.
        return World((MovingObstacle("pedestrian", 13.0 + jitter, -4.4, 0.0, 0.58),))
    if name == "two_targets":
        return World((MovingObstacle("pedestrian", 11.5 + jitter, -4.2, 0.0, 0.65),
                      MovingObstacle("cyclist", 20.5 - jitter, 3.6, -0.35, -0.25, 0.55)))
    if name == "empty":
        return World(())
    if name == "narrow":
        return World((MovingObstacle("pedestrian", 12.0 + jitter, -3.4, 0.0, 0.52),),
                     road_half_width=2.1)
    raise ValueError(f"unknown scenario: {name}")
