"""One-probe viability screening for *strictly feasible* two-stage actions.

This is a counterfactual experiment, NOT a recursive-feasibility certificate.
We advance the current committed actuator queue after enqueuing one prospective
command, then re-run the strict baseline two-stage search at that later state.
Only the first command of the selected original plan is issued to the plant;
the hypothetical 0.6s command persistence is NOT imposed on the real rollout.

The new filter never accepts a candidate rejected by any original hard check.
If no shortlisted first command passes, the baseline safe choice is returned
with a transparent warning rather than fabricating a safe continuation.
"""
from __future__ import annotations

import math
import time

from .model import Command, State, VehicleConfig
from .safety import CommandDelayLine
from .world import World
from .planner import Plan, collision_free
from .two_stage import choose_two_stage


def probe_future(initial: State, command: Command, car: VehicleConfig,
                 cfg, original: CommandDelayLine) -> tuple[list[State], CommandDelayLine]:
    """Predict the actual queued history + one possible outgoing command.

    Mutates a COPY. The caller's real actuator line is unaffected.
    This evaluates a fixed-command hypothetical to obtain a future
    *viability proxy*, not what the rolling controller will necessarily do.
    """
    predicted_line = original.copy()
    predicted_line.enqueue(initial.time, command)
    predicted_path = predicted_line.advance_path(
        initial, initial.time + cfg.viability_probe_s, car, cfg.safety_dt)
    return predicted_path, predicted_line


def choose_viability_probed(initial: State, world: World, car: VehicleConfig,
                            cfg, line: CommandDelayLine) -> Plan:
    started = time.perf_counter()
    if (not isinstance(cfg.viability_top_k, int)
            or isinstance(cfg.viability_top_k, bool)
            or cfg.viability_top_k < 1):
        raise ValueError("viability_top_k must be a positive integer")
    if (not math.isfinite(cfg.viability_probe_s)
            or cfg.viability_probe_s <= cfg.control_delay_s):
        raise ValueError("viability_probe_s must exceed the actuator delay")
    if abs(line.delay_s - cfg.control_delay_s) > 1e-8:
        raise ValueError("planner and actuator delays disagree")

    baseline = choose_two_stage(initial, world, car, cfg, line)
    baseline.cost_parts = dict(baseline.cost_parts)
    baseline.cost_parts.update({
        "lookahead_enabled": 1,
        "lookahead_probed": 0,
        "lookahead_viable": 0,
        "lookahead_change": 0,
        "lookahead_no_viable_shortlist": 0,
        "lookahead_probe_s": cfg.viability_probe_s,
    })
    if baseline.unsafe_fallback or not baseline.feasible_count:
        baseline.cost_parts["lookahead_skipped_unsafe_baseline"] = 1
        baseline.compute_ms = (time.perf_counter() - started) * 1000
        return baseline

    eligible = sorted(
        (r for r in baseline.diagnostics
         if r["status"] in ("eligible", "selected")
         and r.get("continuation_count", 0) > 0
         and math.isfinite(r["total_cost"])),
        key=lambda r: (r["total_cost"], r["target_speed"], r["target_steer"])
    )[:cfg.viability_top_k]
    accepted = None
    summary = []
    for row in eligible:
        command = Command(row["target_speed"], row["target_steer"])
        future, future_line = probe_future(initial, command, car, cfg, line)
        rejected_during_probe = not collision_free(
            future, world, car, predict=True,
            padding=cfg.prediction_padding)
        if rejected_during_probe:
            child_feasible = 0
        else:
            child_plan = choose_two_stage(future[-1], world, car, cfg, future_line)
            child_feasible = (child_plan.feasible_count
                              if not child_plan.unsafe_fallback else 0)
        summary.append((command, child_feasible, rejected_during_probe))
        if child_feasible:
            # Candidates are sorted by original cost.  The first viable one
            # must be selected; later probes cannot change the issued action.
            # This is an exact decision-preserving early exit, not a relaxation
            # of any trajectory or physical-safety check.
            accepted = (command, child_feasible)
            break

    baseline.cost_parts["lookahead_unprobed"] = len(eligible) - len(summary)
    baseline.cost_parts["lookahead_early_exit"] = int(accepted is not None and len(summary) < len(eligible))
    baseline.cost_parts["lookahead_probed"] = len(summary)
    baseline.cost_parts["lookahead_viable"] = sum(bool(count) for _, count, _ in summary)
    baseline.cost_parts["lookahead_probe_path_rejected"] = sum(bad for _, _, bad in summary)

    if accepted is None:
        baseline.cost_parts["lookahead_no_viable_shortlist"] = 1
        baseline.compute_ms = (time.perf_counter() - started) * 1000
        return baseline

    selected_command, selected_child_count = accepted
    baseline.cost_parts["lookahead_child_feasible_count"] = selected_child_count
    if selected_command != baseline.command:
        # Re-run ONLY the strictly feasible first-action subset to recover its
        # best second action and full checked physical brake/hold trajectory.
        alternative = choose_two_stage(
            initial, world, car, cfg, line, forced_first=selected_command)
        if alternative.unsafe_fallback or alternative.feasible_count == 0:
            raise AssertionError("shortlist action lost original strict feasibility")
        baseline.command = alternative.command
        baseline.trajectory = alternative.trajectory
        baseline.score = alternative.score
        baseline.cost_parts.update(alternative.cost_parts)
        baseline.cost_parts["lookahead_change"] = 1
        for row in baseline.diagnostics:
            row["selected"] = (row["target_speed"] == selected_command.target_speed
                               and row["target_steer"] == selected_command.target_steer)
            if row["selected"]:
                row["status"] = "selected"
            elif row["status"] == "selected":
                row["status"] = "eligible"
    baseline.compute_ms = (time.perf_counter() - started) * 1000
    return baseline
