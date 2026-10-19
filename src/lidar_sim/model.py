"""Low-speed kinematic bicycle model with bounded actuation."""
from __future__ import annotations

from dataclasses import dataclass
import math


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class VehicleConfig:
    wheelbase: float = 1.35
    length: float = 1.9
    width: float = 0.85
    max_speed: float = 2.4
    acceleration: float = 0.85
    braking: float = 1.6
    max_steer: float = 0.42
    max_steer_rate: float = 0.75
    # The vehicle's actual length/width never change; this selects only
    # the geometric ROAD boundary approximation, not obstacle collision.
    road_footprint: str = "disc"

    @property
    def collision_radius(self) -> float:
        # Conservative disc covering a body-sized centered rectangle.
        return math.hypot(self.length / 2, self.width / 2)


@dataclass(frozen=True)
class State:
    x: float
    y: float
    yaw: float
    speed: float
    steer: float = 0.0
    time: float = 0.0


@dataclass(frozen=True)
class Command:
    target_speed: float
    target_steer: float


def step(state: State, command: Command, cfg: VehicleConfig, dt: float) -> State:
    if dt <= 0:
        raise ValueError("dt must be positive")
    desired_v = clamp(command.target_speed, 0.0, cfg.max_speed)
    desired_delta = clamp(command.target_steer, -cfg.max_steer, cfg.max_steer)
    dv = clamp(desired_v - state.speed, -cfg.braking * dt, cfg.acceleration * dt)
    v = clamp(state.speed + dv, 0.0, cfg.max_speed)
    steer = state.steer + clamp(desired_delta - state.steer, -cfg.max_steer_rate * dt, cfg.max_steer_rate * dt)
    yaw_rate = v * math.tan(steer) / cfg.wheelbase
    yaw_mid = state.yaw + yaw_rate * dt / 2
    return State(state.x + v * math.cos(yaw_mid) * dt,
                 state.y + v * math.sin(yaw_mid) * dt,
                 (state.yaw + yaw_rate * dt + math.pi) % (2 * math.pi) - math.pi,
                 v, steer, state.time + dt)
