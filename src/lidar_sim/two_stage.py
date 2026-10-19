"""Finite-depth two-stage Ackermann action search with strict backup viability.

An ordered pair of constant target setpoints is a *planning hypothesis*.
Only the FIRST command is submitted to the physical delay queue. On the next
control tick the planner replans; the second planned command is not guaranteed
to execute. Accordingly this is NOT recursive-feasibility proof.
"""
from __future__ import annotations

import math
import time

from .decision import trajectory_cost_breakdown
from .model import Command, State, VehicleConfig
from .safety import CommandDelayLine, braking_tail, stationary_hold
from .world import World
from .planner import Plan, collision_free, rollout


def continuation_options(first: Command, car: VehicleConfig) -> tuple[Command, ...]:
    """Few physically meaningful second actions; preserve strict bounds."""
    options = (
        first,
        Command(0.0, first.target_steer),
        Command(car.max_speed, first.target_steer),
        Command(min(1.2, car.max_speed), 0.0),
        Command(min(1.2, car.max_speed), -first.target_steer),
    )
    # Deduplicate speed/steer pairs deterministically.
    return tuple(dict.fromkeys(options))


def choose_two_stage(initial: State, world: World, car: VehicleConfig, cfg,
                     line: CommandDelayLine, *,
                     committed_progress: bool = False) -> Plan:
    start = time.perf_counter()
    if cfg.two_stage_second_s <= 0:
        raise ValueError("second-stage window must be positive")
    if cfg.first_stage_progress_weight < 0:
        raise ValueError("first-stage progress weight must be nonnegative")
    if cfg.stationary_hold_s < 0:
        raise ValueError("post-stop horizon must be nonnegative")
    padding = cfg.prediction_padding
    prefix = line.copy().advance_path(
        initial, initial.time + cfg.control_delay_s, car, cfg.safety_dt)
    diagnostics: list[dict] = []
    best = None
    feasible_pairs = 0
    attempted_pairs = 0
    rejects = 0

    # The committed prefix is identical for all candidate actions and cannot
    # be cancelled. A collision here is not resolved by a better future score.
    if not collision_free(prefix, world, car, predict=True, padding=padding):
        cmd = Command(0.0, initial.steer)
        return Plan(cmd, rollout(initial, cmd, car, cfg.horizon, cfg.sample_dt),
                    math.inf, 0, 0, (time.perf_counter()-start)*1000,
                    emergency=True, backup_checked=True, unsafe_fallback=True,
                    fallback_strategy="unsafe_committed_prefix")

    for first_v in cfg.speed_samples:
        for first_delta in cfg.steer_samples:
            first = Command(first_v, first_delta)
            row = {"target_speed": first_v, "target_steer": first_delta,
                   "status": "nominal_rejected", "selected": False,
                   "total_cost": float("inf"), "continuation_count": 0}
            diagnostics.append(row)
            first_path = rollout(prefix[-1], first, car,
                                 cfg.candidate_execute_s, cfg.safety_dt)
            if not collision_free(first_path, world, car, predict=True, padding=padding):
                continue
            for second in continuation_options(first, car):
                attempted_pairs += 1
                second_path = rollout(first_path[-1], second, car,
                                      cfg.two_stage_second_s, cfg.safety_dt)
                if not collision_free(second_path, world, car,
                                      predict=True, padding=padding):
                    continue
                brake = braking_tail(second_path[-1], car, cfg.safety_dt)
                if not collision_free(brake, world, car, predict=True, padding=padding):
                    rejects += 1
                    continue
                hold = stationary_hold(brake[-1], cfg.stationary_hold_s,
                                       cfg.safety_dt)
                if not collision_free(hold, world, car, predict=True, padding=padding):
                    rejects += 1
                    continue
                feasible_pairs += 1
                row["continuation_count"] += 1
                controlled = prefix + first_path[1:] + second_path[1:]
                costs = trajectory_cost_breakdown(
                    controlled, first, initial, world, car, cfg,
                    predict=True, padding=padding)
                switch = (cfg.speed_change_weight *
                          (second.target_speed - first.target_speed)**2 +
                          cfg.steer_weight *
                          (second.target_steer - first.target_steer)**2)
                # A standard receding-horizon planner can repeatedly plan
                # "move later" while actually sending stop NOW. Discounting
                # imagined second-stage motion is unnecessary: give an extra
                # reward for progress attainable within the *first* phase.
                first_progress = max(
                    0.0, min(world.goal_x, first_path[-1].x)
                    - min(world.goal_x, prefix[-1].x))
                second_progress = max(
                    0.0, min(world.goal_x, second_path[-1].x)
                    - min(world.goal_x, first_path[-1].x))
                commitment = (-cfg.first_stage_progress_weight * first_progress
                              if committed_progress else 0.0)
                score = costs.total + switch + commitment
                if score < row["total_cost"]:
                    row.update(costs.as_record())
                    row["cost_action_switch"] = switch
                    row["cost_first_stage_commitment"] = commitment
                    row["planned_first_progress_m"] = first_progress
                    row["planned_second_progress_m"] = second_progress
                    row["planned_second_speed"] = second.target_speed
                    row["planned_second_steer"] = second.target_steer
                    row["total_cost"] = score
                    row["second_speed"] = second.target_speed
                    row["second_steer"] = second.target_steer
                    row["status"] = "eligible"
                if best is None or score < best[0]:
                    full = controlled + brake[1:] + hold[1:]
                    best = (score, first, second, full, costs, switch,
                            commitment, first_progress, second_progress)
            if row["continuation_count"] == 0:
                row["status"] = "backup_rejected"

    elapsed = (time.perf_counter()-start)*1000
    if best is None:
        # Strictly report infeasibility, do not silently bypass hard checks.
        cmd = Command(0.0, initial.steer)
        return Plan(cmd, rollout(initial, cmd, car, cfg.horizon, cfg.sample_dt),
                    math.inf, 0, attempted_pairs, elapsed,
                    emergency=True, backup_checked=True,
                    rejected_by_backup=rejects,
                    unsafe_fallback=True, fallback_strategy="emergency_brake",
                    diagnostics=diagnostics)
    score, first, second, full, costs, switch, commitment, first_progress, second_progress = best
    for row in diagnostics:
        if row["target_speed"] == first.target_speed and row["target_steer"] == first.target_steer:
            row["selected"] = True
            row["status"] = "selected"
            break
    cost_parts = costs.as_record()
    cost_parts["cost_action_switch"] = switch
    cost_parts["cost_first_stage_commitment"] = commitment
    cost_parts["planned_first_progress_m"] = first_progress
    cost_parts["planned_second_progress_m"] = second_progress
    cost_parts["planned_second_speed"] = second.target_speed
    cost_parts["planned_second_steer"] = second.target_steer
    cost_parts["total_cost"] = score
    return Plan(first, full, score, feasible_pairs, attempted_pairs, elapsed,
                backup_checked=True, rejected_by_backup=rejects,
                cost_parts=cost_parts, diagnostics=diagnostics)
