"""Explicit terminal success semantics for an Ackermann road corridor.

x_gate is the historical reference-point threshold used in earlier studies.
pose_gate requires the PHYSICAL body center to cross the finish line while
remaining inside a lateral corridor and nearly aligned with the road. A speed
limit is optional: crossing the goal section is not necessarily parking.
"""
from __future__ import annotations

import math

from .model import State, VehicleConfig
from .world import World


def goal_metrics(state: State, world: World, vehicle: VehicleConfig,
                 *, distance_tolerance_m: float = .6,
                 lateral_tolerance_m: float = .75,
                 heading_tolerance_deg: float = 15.,
                 speed_limit_mps: float | None = None) -> dict:
    if (distance_tolerance_m < 0 or lateral_tolerance_m < 0
            or heading_tolerance_deg < 0 or heading_tolerance_deg > 180
            or (speed_limit_mps is not None and speed_limit_mps < 0)):
        raise ValueError("goal tolerances must be nonnegative and heading <= 180 degrees")
    center_x, center_y = vehicle.body_center(state)
    yaw_deg = abs(math.degrees(math.atan2(math.sin(state.yaw), math.cos(state.yaw))))
    line_met = state.x >= world.goal_x - distance_tolerance_m
    center_line_met = center_x >= world.goal_x - distance_tolerance_m
    lateral_met = abs(center_y) <= lateral_tolerance_m
    yaw_met = yaw_deg <= heading_tolerance_deg
    speed_met = speed_limit_mps is None or state.speed <= speed_limit_mps
    return {
        "historical_x_line_met": line_met,
        "center_x_line_met": center_line_met,
        "lateral_met": lateral_met,
        "heading_met": yaw_met,
        "speed_met": speed_met,
        "pose_gate_met": center_line_met and lateral_met and yaw_met and speed_met,
        "body_center_x_m": center_x,
        "body_center_y_m": center_y,
        "abs_heading_error_deg": yaw_deg,
    }
