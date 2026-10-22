"""Strict *existence* witness for one subsequent two-stage plan.

This has exactly the hard-check ordering of choose_two_stage: committed prefix,
first action, second action, physical braking tail, stationary occupancy.
It does not compute ranking scores because only existence is queried. It
returns at the first fully checked witness, not a trajectory to execute.
Any change in physics checks must be validated against choose_two_stage.
"""
from __future__ import annotations

from .model import State, VehicleConfig
from .safety import CommandDelayLine, braking_tail, stationary_hold
from .world import World
from .planner import collision_free, rollout
from .model import Command
from .two_stage import continuation_options


def strict_continuation_exists(initial: State, world: World,
                               car: VehicleConfig, cfg,
                               line: CommandDelayLine) -> bool:
    if cfg.two_stage_second_s <= 0:
        raise ValueError("second-stage window must be positive")
    if cfg.stationary_hold_s < 0:
        raise ValueError("post-stop horizon must be nonnegative")
    padding = cfg.prediction_padding
    prefix = line.copy().advance_path(
        initial, initial.time + cfg.control_delay_s, car, cfg.safety_dt)
    if not collision_free(prefix, world, car, predict=True, padding=padding):
        return False

    for first_v in cfg.speed_samples:
        for first_delta in cfg.steer_samples:
            first = Command(first_v, first_delta)
            first_path = rollout(prefix[-1], first, car,
                                 cfg.candidate_execute_s, cfg.safety_dt)
            if not collision_free(first_path, world, car,
                                  predict=True, padding=padding):
                continue
            for second in continuation_options(first, car):
                second_path = rollout(first_path[-1], second, car,
                                      cfg.two_stage_second_s, cfg.safety_dt)
                if not collision_free(second_path, world, car,
                                      predict=True, padding=padding):
                    continue
                brake = braking_tail(second_path[-1], car, cfg.safety_dt)
                if not collision_free(brake, world, car,
                                      predict=True, padding=padding):
                    continue
                hold = stationary_hold(brake[-1], cfg.stationary_hold_s,
                                       cfg.safety_dt)
                if collision_free(hold, world, car, predict=True,
                                  padding=padding):
                    return True
    return False
