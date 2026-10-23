"""Timestamped tracker -> planner adapter and read-only offline shadow replay.

This module consumes *estimated* dynamic tracks expressed in a fixed local
odom frame. It intentionally does not read LiDAR packets, invent detections,
inject ground truth, publish ROS commands, or score collision outcomes.

For each track, its (x, y) is the estimate AT observed_at_s, not at the
planning tick. Its CV prediction is projected to absolute planner time by
MovingObstacle.position(t), preventing the common double-aging error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping

from .model import Command, State, VehicleConfig
from .planner import PlannerConfig, choose_action
from .safety import CommandDelayLine, ScheduledCommand
from .world import MovingObstacle, World

DEFAULT_MAX_AGE_S = 0.35
SUPPORTED_FRAME = "odom"
SHADOW_MODE = "two_stage_viability"


class SnapshotError(ValueError):
    """Input cannot be interpreted without guessing timestamp or frame."""


def finite_number(data: Mapping[str, Any], key: str) -> float:
    try:
        value = data[key]
    except (KeyError, TypeError) as exc:
        raise SnapshotError(f"missing required {key}") from exc
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise SnapshotError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise SnapshotError(f"{key} must be finite")
    return number


def obj(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    try:
        value = data[key]
    except (KeyError, TypeError) as exc:
        raise SnapshotError(f"missing required {key}") from exc
    if not isinstance(value, dict):
        raise SnapshotError(f"{key} must be an object")
    return value


def command(data: Mapping[str, Any]) -> Command:
    return Command(
        finite_number(data, "target_speed_mps"),
        finite_number(data, "target_steer_rad"),
    )


def enforce_limits(cmd: Command, car: VehicleConfig) -> None:
    if not 0 <= cmd.target_speed <= car.max_speed:
        raise SnapshotError("command speed outside vehicle bounds")
    if abs(cmd.target_steer) > car.max_steer:
        raise SnapshotError("command steer outside vehicle bounds")


@dataclass(frozen=True)
class EstimateSnapshot:
    time_s: float
    state: State
    world: World
    actuator: CommandDelayLine
    max_track_age_s: float
    oldest_track_age_s: float
    track_count: int
    stale_track_ids: tuple[str, ...]

    @property
    def blocked_reason(self) -> str | None:
        if self.track_count == 0:
            return "no_observed_tracks_unverified_free_space"
        if self.stale_track_ids:
            return "stale_track_estimates"
        return None


def parse_snapshot(data: Mapping[str, Any], *,
                   max_age_s: float = DEFAULT_MAX_AGE_S,
                   vehicle: VehicleConfig | None = None) -> EstimateSnapshot:
    """Validate a snapshot with explicit *past* observations and actuator state.

    Missing tracks / stale tracks mean 'do not issue a shadow recommendation';
    they are never converted into an obstacle-free world.
    """
    if not math.isfinite(max_age_s) or max_age_s < 0:
        raise SnapshotError("max_age_s must be finite and nonnegative")
    if not isinstance(data, dict):
        raise SnapshotError("snapshot must be a JSON object")
    if data.get("frame") != SUPPORTED_FRAME:
        raise SnapshotError("frame must be odom (fixed, time-consistent local frame)")
    t = finite_number(data, "t_s")
    if t < 0:
        raise SnapshotError("negative planning time")
    car = vehicle or VehicleConfig(road_footprint="rearaxle_rectangle")
    ego = obj(data, "ego")
    state = State(
        finite_number(ego, "x_m"), finite_number(ego, "y_m"),
        finite_number(ego, "yaw_rad"), finite_number(ego, "speed_mps"),
        finite_number(ego, "steer_rad"), t)
    if not 0 <= state.speed <= car.max_speed:
        raise SnapshotError("ego speed outside vehicle bounds")
    if abs(state.steer) > car.max_steer:
        raise SnapshotError("ego steer outside vehicle bounds")

    actual = obj(data, "actuator")
    delay = finite_number(actual, "delay_s")
    if delay < 0:
        raise SnapshotError("negative actuator delay")
    active = command(obj(actual, "active"))
    enforce_limits(active, car)
    try:
        queued = actual["pending"]
    except KeyError as exc:
        raise SnapshotError("missing actuator.pending; cannot reconstruct delay prefix") from exc
    if not isinstance(queued, list):
        raise SnapshotError("actuator.pending must be a list")
    pending = []
    last_activation = t
    for item in queued:
        if not isinstance(item, dict):
            raise SnapshotError("queued command must be an object")
        when = finite_number(item, "activate_at_s")
        queued_command = command(item)
        enforce_limits(queued_command, car)
        if when <= t + 1e-9:
            raise SnapshotError("past/current actuator activation must already be applied")
        if when < last_activation:
            raise SnapshotError("pending actuator events must be ordered")
        last_activation = when
        pending.append(ScheduledCommand(when, queued_command))
    actuator = CommandDelayLine(active, delay, pending)

    road = obj(data, "road")
    half_width = finite_number(road, "half_width_m")
    goal_x = finite_number(road, "goal_x_m")
    margin = finite_number(road, "safety_margin_m")
    if half_width <= 0 or margin < 0:
        raise SnapshotError("invalid road geometry")
    try:
        tracks = data["tracks"]
    except KeyError as exc:
        raise SnapshotError("missing tracks; cannot assume free space") from exc
    if not isinstance(tracks, list):
        raise SnapshotError("tracks must be an array")
    observed = []
    stale = []
    identities = set()
    oldest = 0.0
    for item in tracks:
        if not isinstance(item, dict):
            raise SnapshotError("track must be an object")
        name = item.get("id")
        if not isinstance(name, str) or not name or name in identities:
            raise SnapshotError("track IDs must be unique nonempty strings")
        identities.add(name)
        observed_at = finite_number(item, "observed_at_s")
        age = t - observed_at
        if age < -1e-9 or observed_at < 0:
            raise SnapshotError("future/negative track observation timestamp")
        oldest = max(oldest, age)
        if age > max_age_s + 1e-9:
            stale.append(name)
        px, py = finite_number(item, "x_m"), finite_number(item, "y_m")
        vx, vy = finite_number(item, "vx_mps"), finite_number(item, "vy_mps")
        radius = finite_number(item, "radius_m")
        if radius <= 0:
            raise SnapshotError("nonpositive obstacle radius")
        # Optional uncertainty values are recorded upstream but not applied
        # as calibrated collision envelopes in this iteration. Reject invalid
        # metadata instead of quietly treating a known covariance as certainty.
        for optional in ("sigma_x_m", "sigma_y_m"):
            if optional in item and finite_number(item, optional) < 0:
                raise SnapshotError(f"{optional} must be nonnegative")
        if "point_count" in item:
            n = item["point_count"]
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise SnapshotError("point_count must be nonnegative integer")
        # MovingObstacle coordinates are model x/y intercepts at absolute t=0.
        observed.append(MovingObstacle(
            name, px - vx * observed_at, py - vy * observed_at,
            vx, vy, radius))
    world = World(tuple(observed), road_half_width=half_width,
                  goal_x=goal_x, safety_margin=margin)
    return EstimateSnapshot(
        t, state, world, actuator, max_age_s, oldest, len(tracks),
        tuple(stale),
    )


def shadow_decision(snapshot: EstimateSnapshot, *,
                    planner: PlannerConfig | None = None,
                    vehicle: VehicleConfig | None = None) -> dict[str, Any]:
    """Evaluate a recorded state, never enqueue or publish the result."""
    record = {
        "time_s": snapshot.time_s,
        "frame": SUPPORTED_FRAME,
        "track_count": snapshot.track_count,
        "oldest_track_age_s": snapshot.oldest_track_age_s,
        "max_allowed_track_age_s": snapshot.max_track_age_s,
        "stale_track_ids": "|".join(snapshot.stale_track_ids),
        "mode": SHADOW_MODE,
        "status": "",
        "recommended_speed_mps": "",
        "recommended_steer_rad": "",
        "strict_feasible_first_actions": "",
        "unsafe_fallback": "",
        "planner_wall_ms": "",
        "shadow_only": 1,
    }
    if snapshot.blocked_reason:
        record["status"] = snapshot.blocked_reason
        return record
    car = vehicle or VehicleConfig(road_footprint="rearaxle_rectangle")
    cfg = planner or PlannerConfig()
    cfg = replace(cfg, control_delay_s=snapshot.actuator.delay_s)
    plan = choose_action(snapshot.state, snapshot.world, car, cfg,
                         mode=SHADOW_MODE, delay_line=snapshot.actuator.copy())
    record.update(
        status=("unsafe_fallback_no_guaranteed_action"
                if plan.unsafe_fallback else "evaluated_not_safety_certified"),
        recommended_speed_mps=plan.command.target_speed,
        recommended_steer_rad=plan.command.target_steer,
        strict_feasible_first_actions=sum(
            d.get("status") in ("selected", "eligible")
            and d.get("continuation_count", 0) > 0
            for d in plan.diagnostics),
        unsafe_fallback=int(plan.unsafe_fallback),
        planner_wall_ms=plan.compute_ms,
    )
    return record


def read_snapshots(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """JSONL is explicit per-decision state; no ground-truth episode replay."""
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SnapshotError(f"line {line_no}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise SnapshotError(f"line {line_no}: snapshot must be object")
            yield line_no, item


def evaluate_file(path: Path, *, max_age_s: float = DEFAULT_MAX_AGE_S,
                  max_gap_s: float = 0.50) -> list[dict[str, Any]]:
    """Fail if frame sequence or timestamps are broken; no hidden interpolation."""
    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise SnapshotError("max_gap_s must be positive and finite")
    results = []
    last_time = None
    for line_no, item in read_snapshots(path):
        try:
            snap = parse_snapshot(item, max_age_s=max_age_s)
        except SnapshotError as exc:
            raise SnapshotError(f"line {line_no}: {exc}") from exc
        if last_time is not None and snap.time_s <= last_time:
            raise SnapshotError(f"line {line_no}: timestamps not strictly increasing")
        if last_time is not None and snap.time_s - last_time > max_gap_s:
            # A gap is a failed data acquisition; do not interpolate to hide it.
            results.append({
                "time_s": snap.time_s, "frame": SUPPORTED_FRAME,
                "track_count": snap.track_count,
                "oldest_track_age_s": snap.oldest_track_age_s,
                "max_allowed_track_age_s": max_age_s,
                "stale_track_ids": "|".join(snap.stale_track_ids),
                "mode": SHADOW_MODE, "status": "timestamp_gap_no_recommendation",
                "recommended_speed_mps": "", "recommended_steer_rad": "",
                "strict_feasible_first_actions": "", "unsafe_fallback": "",
                "planner_wall_ms": "", "shadow_only": 1,
            })
        else:
            results.append(shadow_decision(snap))
        last_time = snap.time_s
    if not results:
        raise SnapshotError("no snapshots in recording")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only offline estimated-track planning; no ROS output")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-track-age", type=float,
                        default=DEFAULT_MAX_AGE_S)
    parser.add_argument("--max-gap", type=float, default=.50)
    args = parser.parse_args()
    rows = evaluate_file(args.input, max_age_s=args.max_track_age,
                         max_gap_s=args.max_gap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        fields = list(rows[0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Read {len(rows)} snapshots; evaluated "
          f"{sum(r['status'].startswith('evaluated') for r in rows)}, "
          f"blocked/gap {sum(not r['status'].startswith('evaluated') for r in rows)}")
    print("Shadow only: no command is sent to an actuator or ROS topic.")


if __name__ == "__main__":
    main()
