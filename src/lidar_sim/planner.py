"""Finite-set candidate planning with time-aligned dynamic occupancy checks."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

from .model import Command, State, VehicleConfig, step
from .world import World
from .safety import CommandDelayLine, backup_segments, braking_then_hold
from .decision import trajectory_cost_breakdown
from .recovery import rank_rescue, risk_of_path


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
    heading_weight: float = 0.4
    clearance_weight: float = 0.6
    clearance_target_m: float = 0.8
    prediction_padding: float = 0.12
    control_delay_s: float = 0.25
    candidate_execute_s: float = 0.40
    safety_dt: float = 0.05
    stationary_hold_s: float = 6.0
    # Recovery uses existing hold horizon; no reduction of hard safety padding.


@dataclass
class Plan:
    command: Command
    trajectory: list[State]
    score: float
    feasible_count: int
    candidate_count: int
    compute_ms: float
    emergency: bool = False
    backup_checked: bool = False
    rejected_by_backup: int = 0
    unsafe_fallback: bool = False
    fallback_strategy: str = "none"
    rescue_immediate_feasible_count: int = 0
    fallback_risk_exposure: float | None = None
    cost_parts: dict = field(default_factory=dict)
    diagnostics: list[dict] = field(default_factory=list)


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
                    world: World, cfg: PlannerConfig,
                    vehicle: VehicleConfig | None = None, *,
                    predict: bool = True, padding: float = 0.0) -> float:
    """Compatibility wrapper around inspectable, weighted objective components."""
    return trajectory_cost_breakdown(path, command, start, world,
                                     vehicle or VehicleConfig(), cfg,
                                     predict=predict, padding=padding).total


def choose_action(initial: State, world: World, vehicle: VehicleConfig,
                  cfg: PlannerConfig, *, mode: str = "predictive",
                  delay_line: CommandDelayLine | None = None) -> Plan:
    """Select safe candidate using a transparent finite action set.

    reactive: freeze obstacle positions at observation time.
    predictive: CV moving obstacles, but planner assumes instant actuation.
    delay_aware: use actual already-queued commands during actuator delay.
    braking: delay_aware plus execute-window/backup-braking tail filter.
    braking_hold: braking plus explicit occupancy check *after* the car stops
                  for stationary_hold_s seconds. Neither mode is formally safe.
    recovery: identical nominal/backup hard checks to braking_hold; ONLY when
              there are no certified candidates, ranks the 0.4s-immediate-safe
              actions by forecast violation risk. This is an UNSAFE fallback.

    All modes have the same *actual* actuator delay in closed-loop experiments.
    Future versions will replace the ideal target state with radar tracking.
    """
    if mode not in ("reactive", "predictive", "delay_aware", "braking", "braking_hold", "recovery"):
        raise ValueError(f"unknown planner mode: {mode}")
    start_time = time.perf_counter()
    if delay_line is None:
        delay_line = CommandDelayLine(Command(initial.speed, initial.steer), cfg.control_delay_s)
    if abs(delay_line.delay_s - cfg.control_delay_s) > 1e-8:
        raise ValueError("planner/actuator control delays must agree")
    prefix = None
    if mode in ("delay_aware", "braking", "braking_hold", "recovery"):
        # Use the *queued* historical commands, not just the most recent one.
        prefix = delay_line.copy().advance_path(
            initial, initial.time + cfg.control_delay_s, vehicle, cfg.safety_dt)
    best = None
    best_nominal = None
    diagnostics: list[dict] = []
    rescue_options: list[tuple] = []
    count = 0
    total = 0
    backup_rejected = 0
    for target_speed in cfg.speed_samples:
        for target_steer in cfg.steer_samples:
            total += 1
            command = Command(target_speed, target_steer)
            if prefix is None:
                path = rollout(initial, command, vehicle, cfg.horizon, cfg.sample_dt)
            else:
                # Candidate starts only after previously transmitted commands finish.
                candidate = rollout(prefix[-1], command, vehicle, cfg.horizon, cfg.sample_dt)
                path = prefix[:-1] + candidate
            # Equal padding across modes: freezing targets is the actual
            # reactive baseline difference, not secretly reducing safety margin.
            padding = cfg.prediction_padding
            predicted = mode != "reactive"
            costs = trajectory_cost_breakdown(
                path, command, initial, world, vehicle, cfg,
                predict=predicted, padding=padding)
            score = costs.total
            info = {"target_speed": target_speed, "target_steer": target_steer,
                    **costs.as_record(), "status": "eligible", "selected": False}
            diagnostics.append(info)
            nominal_clear = collision_free(path, world, vehicle,
                                           predict=predicted, padding=padding)
            segs = None
            if mode == "recovery":
                # Do not allow an unverified fallback to break the genuinely
                # committed prefix or the short executable control segment.
                segs = backup_segments(initial, command, delay_line, vehicle,
                                       cfg.candidate_execute_s, cfg.safety_dt)
                immediate = segs.prefix + segs.candidate[1:]
                immediate_clear = collision_free(immediate, world, vehicle,
                                                  predict=True, padding=padding)
                info["immediate_safe"] = immediate_clear
                if immediate_clear:
                    hold_path = braking_then_hold(
                        segs, cfg.stationary_hold_s, cfg.safety_dt)
                    hold_risk = risk_of_path(hold_path, world, vehicle, padding=padding)
                    nominal_risk = risk_of_path(path, world, vehicle, padding=padding)
                    # Risk is assessed over both the predicted nominal rollout
                    # and the backup-to-stop/hold contingency. Scoring does not
                    # relax any nominal or brake hard constraint in normal mode.
                    risk_value = hold_risk.violation_exposure + nominal_risk.violation_exposure
                    from .recovery import Risk
                    risk = Risk(min(hold_risk.minimum_clearance_m,
                                    nominal_risk.minimum_clearance_m),
                                risk_value,
                                min(hold_risk.first_violation_after_s,
                                    nominal_risk.first_violation_after_s))
                    rank = rank_rescue(nominal_collision_free=nominal_clear,
                                       risk=risk, nominal_cost=score)
                    info.update({"rescue_violation_exposure": risk.violation_exposure,
                                 "rescue_min_clearance_m": risk.minimum_clearance_m,
                                 "rescue_first_violation_s": risk.first_violation_after_s})
                    rescue_options.append((rank, score, command, path, costs, risk))
            if not nominal_clear:
                info["status"] = "nominal_rejected"
                continue
            if best_nominal is None or score < best_nominal[0]:
                best_nominal = (score, command, path, costs)
            if mode in ("braking", "braking_hold", "recovery"):
                if segs is None:
                    segs = backup_segments(initial, command, delay_line, vehicle,
                                           cfg.candidate_execute_s, cfg.safety_dt)
                if mode in ("braking_hold", "recovery"):
                    future_path = braking_then_hold(
                        segs, cfg.stationary_hold_s, cfg.safety_dt)
                else:
                    future_path = segs.complete
                if not collision_free(future_path, world, vehicle,
                                      predict=True, padding=padding):
                    backup_rejected += 1
                    info["status"] = "backup_rejected"
                    continue
            count += 1
            if best is None or score < best[0]:
                best = (score, command, path, costs)
    elapsed = (time.perf_counter() - start_time) * 1000

    def report_selection(entry, status: str) -> dict:
        cmd = entry[1]
        for row in diagnostics:
            if row["target_speed"] == cmd.target_speed and row["target_steer"] == cmd.target_steer:
                row["selected"] = True
                row["status"] = status
                break
        return entry[3].as_record()
    if best is None and mode == "recovery" and rescue_options:
        # Last-resort ranking, only over the short-execution-safe action set.
        # Report it as unverified: the future scenario may still be unsafe.
        rank, score, command, path, costs, risk = min(rescue_options, key=lambda item: item[0])
        parts = report_selection((score, command, path, costs),
                                 "unsafe_risk_ranked_fallback")
        return Plan(command, path, score, 0, total, elapsed, backup_checked=True,
                    rejected_by_backup=backup_rejected, unsafe_fallback=True,
                    fallback_strategy="risk_ranked_no_backup", cost_parts=parts,
                    diagnostics=diagnostics,
                    rescue_immediate_feasible_count=len(rescue_options),
                    fallback_risk_exposure=risk.violation_exposure)
    if best is None and mode in ("braking", "braking_hold") and best_nominal is not None:
        # No candidate has a verified backup. Braking can itself be unsafe
        # (e.g. crossing traffic hits a stopped robot). Continue with the
        # lowest-cost nominally clear candidate, explicitly flagged UNSAFE.
        parts = report_selection(best_nominal, "unsafe_fallback")
        return Plan(best_nominal[1], best_nominal[2], best_nominal[0], 0, total, elapsed,
                    backup_checked=True, rejected_by_backup=backup_rejected,
                    unsafe_fallback=True, fallback_strategy="nominal_no_backup",
                    cost_parts=parts, diagnostics=diagnostics)
    if best is None:
        # Brake request subject to actuator delay/physics; no safety proof.
        # If the committed prefix is unsafe, this code CANNOT guarantee safety.
        emergency = Command(0.0, initial.steer)
        path = rollout(initial, emergency, vehicle, cfg.horizon, cfg.sample_dt)
        return Plan(emergency, path, math.inf, 0, total, elapsed,
                    emergency=True, backup_checked=(mode in ("braking", "braking_hold", "recovery")),
                    rejected_by_backup=backup_rejected,
                    unsafe_fallback=True, fallback_strategy="emergency_brake",
                    diagnostics=diagnostics)
    parts = report_selection(best, "selected")
    return Plan(best[1], best[2], best[0], count, total, elapsed,
                backup_checked=(mode in ("braking", "braking_hold", "recovery")), rejected_by_backup=backup_rejected,
                cost_parts=parts, diagnostics=diagnostics)
