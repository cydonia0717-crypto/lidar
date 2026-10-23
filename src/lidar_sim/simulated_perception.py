"""Causal, configurable 2-D LiDAR-like track observations for simulation.

This is *not* a ray-casting or PointCloud2 model. The simulator samples
ground-truth obstacle centers at PAST capture times, then generates stochastic
detections and estimates velocity only from earlier detections. The planner
receives only estimated tracks. Physical collisions always use a separate
World truth. Source semantics are compatible with the phase18 tracked-shadow
contract, allowing a future ROS adapter to replace this simulated provider.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random

from .model import State
from .world import MovingObstacle, World


@dataclass(frozen=True)
class PerceptionConfig:
    position_sigma_m: float = 0.08
    range_sigma_per_m: float = 0.003
    dropout_probability: float = 0.05
    detection_range_m: float = 45.0
    field_of_view_rad: float = 2 * math.pi
    capture_delay_ticks: int = 1
    sensor_every_ticks: int = 1
    max_track_age_s: float = 0.35
    minimum_updates: int = 2
    velocity_update_gain: float = 0.5

    def __post_init__(self) -> None:
        finite_nonnegative = (
            self.position_sigma_m, self.range_sigma_per_m,
            self.dropout_probability, self.detection_range_m,
            self.field_of_view_rad, self.max_track_age_s,
            self.velocity_update_gain,
        )
        if any(not math.isfinite(x) or x < 0 for x in finite_nonnegative):
            raise ValueError("invalid nonfinite/negative perception parameter")
        if not 0 <= self.dropout_probability <= 1:
            raise ValueError("dropout_probability must lie in [0,1]")
        if self.detection_range_m <= 0:
            raise ValueError("detection_range_m must be positive")
        if not 0 < self.field_of_view_rad <= 2 * math.pi:
            raise ValueError("field_of_view_rad must lie in (0,2pi]")
        if not 0 < self.velocity_update_gain <= 1:
            raise ValueError("velocity_update_gain must lie in (0,1]")
        for field in ("capture_delay_ticks", "sensor_every_ticks", "minimum_updates"):
            value = getattr(self, field)
            lower = 0 if field == "capture_delay_ticks" else (
                1 if field == "sensor_every_ticks" else 2)
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{field} must be integer >= {lower}")


@dataclass(frozen=True)
class Track:
    name: str
    observed_time_s: float
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    radius_m: float
    updates: int


@dataclass(frozen=True)
class PerceptionFrame:
    planning_time_s: float
    last_capture_time_s: float | None
    estimated_world: World
    observation_count: int
    valid_tracks: int
    withheld_tracks: int
    stale_track_ids: tuple[str, ...]
    max_valid_age_s: float
    blocked_reason: str | None


class SimulatedPerception:
    """Seeded per-episode tracker; create a fresh instance for each rollout."""

    def __init__(self, config: PerceptionConfig, seed: int):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.config = config
        self.seed = seed
        self._latest: dict[str, Track] = {}
        self._last_capture_tick = -1
        self._last_capture_time: float | None = None

    def _rng(self, tick: int, name: str) -> random.Random:
        # Stable regardless of Python hash randomization or planner choices.
        digest = hashlib.sha256(
            f"phase18|{self.seed}|{tick}|{name}".encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:16], "big"))

    def observe(self, truth: World, ego_history: list[State], *,
                dt: float) -> PerceptionFrame:
        """Read only captured history and return a completely estimated world.

        ego_history[tick] is the REAL ego at start of that control tick,
        necessary to simulate range/FOV at capture time (not at plan time).
        All future targets are unavailable to the returned planner world.
        """
        if not ego_history or not math.isfinite(dt) or dt <= 0:
            raise ValueError("valid ego_history and dt required")
        tick = len(ego_history) - 1
        now = ego_history[-1].time
        cfg = self.config
        final_tick = tick - cfg.capture_delay_ticks
        captured = 0
        while self._last_capture_tick + 1 <= final_tick:
            capture_tick = self._last_capture_tick + 1
            self._last_capture_tick = capture_tick
            if capture_tick % cfg.sensor_every_ticks:
                continue
            capture_ego = ego_history[capture_tick]
            t_obs = capture_ego.time
            self._last_capture_time = t_obs
            for obj in truth.obstacles:
                ox, oy = obj.position(t_obs)
                dx, dy = ox - capture_ego.x, oy - capture_ego.y
                distance = math.hypot(dx, dy)
                if distance > cfg.detection_range_m:
                    continue
                angle = (math.atan2(dy, dx) - capture_ego.yaw + math.pi) % (
                    2 * math.pi) - math.pi
                if (cfg.field_of_view_rad < 2 * math.pi
                        and abs(angle) > cfg.field_of_view_rad / 2):
                    continue
                rng = self._rng(capture_tick, obj.name)
                if rng.random() < cfg.dropout_probability:
                    continue
                sigma = cfg.position_sigma_m + cfg.range_sigma_per_m * distance
                x = ox + rng.gauss(0, sigma)
                y = oy + rng.gauss(0, sigma)
                previous = self._latest.get(obj.name)
                if previous is None:
                    vx = vy = 0.0
                    updates = 1
                else:
                    elapsed = t_obs - previous.observed_time_s
                    if elapsed <= 0:
                        raise ValueError("observation time not increasing")
                    raw_vx = (x - previous.x_m) / elapsed
                    raw_vy = (y - previous.y_m) / elapsed
                    gain = cfg.velocity_update_gain
                    if previous.updates == 1:
                        vx, vy = raw_vx, raw_vy
                    else:
                        vx = (1 - gain) * previous.vx_mps + gain * raw_vx
                        vy = (1 - gain) * previous.vy_mps + gain * raw_vy
                    updates = previous.updates + 1
                self._latest[obj.name] = Track(
                    obj.name, t_obs, x, y, vx, vy, obj.radius, updates)
                captured += 1
        estimates = []
        stale = []
        withheld = 0
        ages = []
        for name, track in sorted(self._latest.items()):
            age = now - track.observed_time_s
            if age > cfg.max_track_age_s + 1e-9:
                stale.append(name)
                continue
            if track.updates < cfg.minimum_updates:
                withheld += 1
                continue
            ages.append(age)
            estimates.append(MovingObstacle(
                name, track.x_m, track.y_m, track.vx_mps, track.vy_mps,
                track.radius_m, track.observed_time_s))
        # Do not infer free space from zero detections, and don't silently
        # forget stale targets as if their occupied space became free.
        blocked = ("stale_tracks_unresolved" if stale else
                   # A newly visible target without two observations may
                   # already be a collision risk; valid older tracks must
                   # not mask it while its velocity remains unknown.
                   "uninitialized_or_no_tracks"
                   if withheld or not estimates else None)
        perceived = World(tuple(estimates),
                          road_half_width=truth.road_half_width,
                          goal_x=truth.goal_x,
                          safety_margin=truth.safety_margin)
        return PerceptionFrame(
            now, self._last_capture_time, perceived, captured,
            len(estimates), withheld, tuple(stale),
            max(ages) if ages else 0.0, blocked)
