"""Explicit least-risk fallback for unavoidably infeasible backup sets.

This is NOT a control-barrier certificate. It only ranks violations in the
same assumed CV environment when all strictly certified candidates have failed.
A collision can remain unavoidable, especially in committed latency prefix.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .model import State, VehicleConfig
from .world import World


@dataclass(frozen=True)
class Risk:
    minimum_clearance_m: float
    violation_exposure: float
    first_violation_after_s: float

    def as_record(self) -> dict:
        return {"minimum_clearance_m": self.minimum_clearance_m,
                "violation_exposure": self.violation_exposure,
                "first_violation_after_s": self.first_violation_after_s}


def risk_of_path(path: list[State], world: World, vehicle: VehicleConfig,
                 *, padding: float = 0.0) -> Risk:
    """Integrate squared shortfalls of the SAME conservative collision margin.

    Integral uses the actual time steps, so a 6-second hold is not disguised
    by differing sample counts. Infinite first_violation means no violation
    in the finite path; this is not a proof for all future time.
    """
    if not path:
        raise ValueError("path is empty")
    min_clearance = float("inf")
    exposure = 0.0
    first = float("inf")
    for i, state in enumerate(path):
        if i and state.time <= path[i-1].time:
            raise ValueError("path timestamps must strictly increase")
        gap = world.road_clearance(state, vehicle) - world.safety_margin
        for target in world.obstacles:
            ox, oy = target.position(state.time)
            gap = min(gap, math.hypot(state.x-ox, state.y-oy)
                      -vehicle.collision_radius-target.radius-world.safety_margin-padding)
        min_clearance = min(min_clearance, gap)
        if gap < 0:
            if first == float("inf"):
                first = state.time - path[0].time
            dt = state.time - path[i-1].time if i else 0.0
            exposure += gap * gap * dt
    return Risk(min_clearance, exposure, first)


def rank_rescue(*, nominal_collision_free: bool, risk: Risk,
                nominal_cost: float) -> tuple[float, ...]:
    """Lexicographic, transparent preference; all candidates MUST first pass
    the committed-plus-execution hard constraints.
    """
    if not math.isfinite(nominal_cost):
        raise ValueError("candidate cost must be finite")
    # Preserve a full-horizon collision-free preference, rather than relaxing
    # it merely to produce a visually impressive success rate.
    return (0 if nominal_collision_free else 1,
            risk.violation_exposure,
            -risk.first_violation_after_s,
            -risk.minimum_clearance_m,
            nominal_cost)
