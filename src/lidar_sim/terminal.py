"""Soft terminal-stop clearance preference, never a hard safety substitute.

The existing braking_hold filter must accept a trajectory before this score
is considered. This term only ranks *already feasible* stopping locations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .model import State, VehicleConfig
from .safety import stationary_hold
from .world import World


@dataclass(frozen=True)
class TerminalMargin:
    minimum_clearance_m: float
    penalty: float


def terminal_stop_margin(state: State, world: World, vehicle: VehicleConfig,
                         *, hold_s: float, dt: float, padding: float,
                         desired_m: float, weight: float) -> TerminalMargin:
    if state.speed > 1e-7:
        raise ValueError("terminal stop must be stationary")
    if desired_m < 0 or weight < 0:
        raise ValueError("terminal preference must be nonnegative")
    points = stationary_hold(state, hold_s, dt)
    free = float("inf")
    for s in points:
        gap = world.road_clearance(s, vehicle) - world.safety_margin
        for target in world.obstacles:
            ox, oy = target.position(s.time)
            gap = min(gap, math.hypot(s.x-ox, s.y-oy) -
                      vehicle.collision_radius - target.radius -
                      world.safety_margin - padding)
        free = min(free, gap)
    return TerminalMargin(free, weight * max(0.0, desired_m - free)**2)
