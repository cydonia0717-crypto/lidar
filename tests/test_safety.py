import pytest

from lidar_sim.model import Command, State, VehicleConfig
from lidar_sim.planner import PlannerConfig, choose_action, collision_free, rollout
from lidar_sim.safety import CommandDelayLine, backup_segments, braking_tail
from lidar_sim.world import MovingObstacle, World


def test_delay_line_applies_command_only_after_latency():
    car = VehicleConfig()
    line = CommandDelayLine(Command(1.0, 0), delay_s=0.25)
    line.enqueue(0.0, Command(0.0, 0))
    before = line.advance_path(State(0, 0, 0, 1.0), .2, car, .05)
    assert before[-1].speed == pytest.approx(1.0)
    after = line.advance_path(before[-1], .35, car, .05)
    assert after[-1].speed == pytest.approx(.84)
    assert line.active == Command(0.0, 0)


def test_delay_prefix_accounts_for_multiple_already_queued_commands():
    car = VehicleConfig()
    line = CommandDelayLine(Command(1.0, 0), delay_s=.30)
    line.enqueue(-.20, Command(.5, .1))
    line.enqueue(-.10, Command(0, .0))
    segments = backup_segments(State(0, 0, 0, 1.0), Command(2.0, 0), line,
                               car, execute_s=.2, dt=.05)
    assert segments.prefix[-1].time == pytest.approx(.30)
    assert segments.prefix[-1].speed < 1.0
    assert len(line.pending) == 2
    assert line.active == Command(1.0, 0)


def test_brake_respects_physical_deceleration_and_end_speed():
    car = VehicleConfig(braking=1.5)
    state = State(0, 0, 0, 2.0, steer=.2)
    path = braking_tail(state, car, .04)
    assert path[-1].speed == pytest.approx(0.0, abs=1e-9)
    assert path[-1].time == pytest.approx(2.0 / 1.5, abs=1e-8)
    assert path[-1].x > 0
    assert all(a.speed >= b.speed for a, b in zip(path, path[1:]))
    assert path[-1].yaw > 0


def test_safe_nominal_trajectory_can_have_unsafe_braking_tail():
    car = VehicleConfig()
    cfg = PlannerConfig()
    start = State(0, 0, 0, 1.5)
    world = World((MovingObstacle("late-crossing", .5, -4., 0., 2.3),))
    held = CommandDelayLine(Command(1.5, 0), cfg.control_delay_s)
    move = Command(2.4, 0)
    nominal = rollout(start, move, car, cfg.horizon, cfg.sample_dt)
    assert collision_free(nominal, world, car, padding=cfg.prediction_padding)
    parts = backup_segments(start, move, held, car, cfg.candidate_execute_s, cfg.safety_dt)
    assert parts.prefix[-1].time == pytest.approx(cfg.control_delay_s)
    assert parts.candidate[-1].time == pytest.approx(cfg.control_delay_s + cfg.candidate_execute_s)
    assert parts.brake[-1].speed == pytest.approx(0.0)
    assert not collision_free(parts.complete, world, car, padding=cfg.prediction_padding)
    base = choose_action(start, world, car, cfg, mode="delay_aware", delay_line=held)
    filtered = choose_action(start, world, car, cfg, mode="braking", delay_line=held)
    assert base.feasible_count > filtered.feasible_count
    assert filtered.rejected_by_backup > 0
    assert filtered.unsafe_fallback
    assert filtered.fallback_strategy == "nominal_no_backup"


def test_delay_config_inconsistency_is_rejected():
    with pytest.raises(ValueError, match="delays"):
        choose_action(State(0, 0, 0, 1), World(()), VehicleConfig(),
                      PlannerConfig(), mode="braking",
                      delay_line=CommandDelayLine(Command(1, 0), delay_s=.1))


def test_invalid_dynamics_parameters_rejected():
    with pytest.raises(ValueError):
        CommandDelayLine(Command(0, 0), delay_s=-1)
    with pytest.raises(ValueError):
        braking_tail(State(0, 0, 0, 1), VehicleConfig(braking=0), .05)


def test_empty_scene_has_valid_backup():
    start = State(0, 0, 0, 1.0)
    cfg = PlannerConfig()
    plan = choose_action(start, World(()), VehicleConfig(), cfg, mode='braking')
    assert plan.backup_checked
    assert plan.feasible_count > 0
    assert plan.rejected_by_backup == 0
    assert not plan.unsafe_fallback


def test_simulated_delay_is_logged_and_reproducible():
    from lidar_sim.experiment import run
    plan = PlannerConfig(control_delay_s=.50)
    first = run('empty', mode='braking', planner=plan)
    second = run('empty', mode='braking', planner=plan)
    # Measured CPU time is intentionally NOT a deterministic vehicle state.
    assert [{k:v for k,v in row.items() if k != "planning_compute_ms"}
            for row in first.trace] == [
                {k:v for k,v in row.items() if k != "planning_compute_ms"}
                for row in second.trace]
    assert first.control_delay_s == .50
    assert first.success
    assert first.trace[0]['applied_speed'] == pytest.approx(1.5)
    assert first.trace[0]['command_speed'] == pytest.approx(2.4)
