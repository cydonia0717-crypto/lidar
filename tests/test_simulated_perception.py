"""Phase18 simulation: truth/observation separation and deterministic ageing."""
from dataclasses import replace

import pytest

from lidar_sim.experiment import RunConfig, run
from lidar_sim.model import State
from lidar_sim.simulated_perception import (
    PerceptionConfig, SimulatedPerception,
)
from lidar_sim.world import MovingObstacle, World


def history(n, dt=.1):
    return [State(0, 0, 0, 0, time=i*dt) for i in range(n)]


def scene(vy=.6):
    return World((MovingObstacle("person", 13, -4, 0, vy, .35),))


def test_velocity_requires_two_past_samples_and_never_uses_truth_velocity():
    model = SimulatedPerception(PerceptionConfig(
        position_sigma_m=0, range_sigma_per_m=0,
        dropout_probability=0, capture_delay_ticks=1), seed=7)
    initial = model.observe(scene(), history(1), dt=.1)
    assert initial.blocked_reason == "uninitialized_or_no_tracks"
    first = model.observe(scene(), history(2), dt=.1)
    assert first.observation_count == 1
    assert first.withheld_tracks == 1
    assert first.valid_tracks == 0
    assert first.blocked_reason == "uninitialized_or_no_tracks"
    second = model.observe(scene(), history(3), dt=.1)
    assert second.valid_tracks == 1
    obs = second.estimated_world.obstacles[0]
    assert obs.reference_time_s == pytest.approx(.1)
    assert obs.position(.1) == pytest.approx((13., -3.94))
    assert obs.vy == pytest.approx(.6)
    assert obs.position(.2) == pytest.approx((13., -3.88))
    assert second.max_valid_age_s == pytest.approx(.1)


def test_capture_at_zero_does_not_reveal_future_truth_velocity():
    cfg = PerceptionConfig(position_sigma_m=0, range_sigma_per_m=0,
                           capture_delay_ticks=0)
    a = SimulatedPerception(cfg, seed=5).observe(scene(.6), history(1), dt=.1)
    b = SimulatedPerception(cfg, seed=5).observe(scene(-.8), history(1), dt=.1)
    assert a.estimated_world == b.estimated_world
    assert a.withheld_tracks == b.withheld_tracks == 1


def test_deterministic_noise_independent_of_global_rng_and_mode_order():
    cfg = PerceptionConfig(position_sigma_m=.15,
                           range_sigma_per_m=.004, dropout_probability=.2)
    def replay():
        tracker = SimulatedPerception(cfg, 19)
        return [tracker.observe(scene(), history(n), dt=.1)
                for n in range(1, 15)]
    one, two = replay(), replay()
    assert one == two
    assert any(a.valid_tracks > 0 for a in one)
    assert any(a.blocked_reason for a in one)


def test_permanent_dropout_is_never_interpreted_as_free_space():
    tracker = SimulatedPerception(PerceptionConfig(
        dropout_probability=1, capture_delay_ticks=0), seed=99)
    outputs = [tracker.observe(scene(), history(n), dt=.1)
               for n in range(1, 8)]
    assert all(f.blocked_reason == "uninitialized_or_no_tracks"
               and not f.estimated_world.obstacles for f in outputs)


def test_stale_previous_track_blocks_all_planning():
    tracker = SimulatedPerception(PerceptionConfig(
        position_sigma_m=0, range_sigma_per_m=0,
        capture_delay_ticks=0, sensor_every_ticks=10,
        max_track_age_s=.15), seed=3)
    for n in range(1, 4):
        final = tracker.observe(scene(), history(n), dt=.1)
    assert final.stale_track_ids == ("person",)
    assert final.blocked_reason == "stale_tracks_unresolved"


@pytest.mark.parametrize("changes", [
    {"dropout_probability": -0.1}, {"dropout_probability": 1.1},
    {"detection_range_m": 0}, {"sensor_every_ticks": 0},
    {"capture_delay_ticks": -1}, {"minimum_updates": 1},
    {"position_sigma_m": float("nan")}, {"max_track_age_s": -1},
    {"velocity_update_gain": 0},
])
def test_bad_sensor_configs_fail(changes):
    with pytest.raises(ValueError):
        PerceptionConfig(**changes)


def test_real_world_collision_is_not_masked_by_missing_sensor_targets():
    colliding = World((MovingObstacle("physically-present", 0, 0, 0, 0),))
    result = run("empty", mode="two_stage_viability",
                 config=RunConfig(max_time=.2),
                 world_override=colliding,
                 perception=SimulatedPerception(PerceptionConfig(
                     dropout_probability=1), 8))
    assert result.collision
    assert not result.success
    assert result.perception_enabled
    assert result.perception_planned_steps == 0


def test_no_sensor_data_brakes_through_real_actuator_delay_and_logs_failure():
    result = run("crossing", mode="two_stage_viability",
                 config=RunConfig(max_time=.3),
                 perception=SimulatedPerception(PerceptionConfig(
                     dropout_probability=1), 8))
    assert result.perception_enabled
    assert result.perception_blocked_steps > 0
    assert result.perception_planned_steps == 0
    assert all(row["command_speed"] == 0 for row in result.trace[:-1])
    assert all(row["fallback_strategy"] == "perception_untrusted_brake"
               for row in result.trace[:-1])
    assert all(row["perception_source"] == "causal_estimated_tracks"
               for row in result.trace[:-1])
    assert result.trace[0]["applied_speed"] == pytest.approx(1.5)
    # Requesting zero speed cannot cancel already-committed physical motion.


def test_imperfect_tracks_only_enter_planner_not_physical_world(monkeypatch):
    from lidar_sim import experiment
    from lidar_sim.model import Command
    from lidar_sim.planner import Plan

    original = scene()
    seen = []
    def fake_choose(state, planning_world, car, cfg, *, mode, delay_line):
        seen.append(planning_world)
        return Plan(Command(0, 0), [], 0, 1, 1, 0)
    monkeypatch.setattr(experiment, "choose_action", fake_choose)
    result = run("crossing", world_override=original,
                 config=RunConfig(max_time=.4, start_speed=0),
                 perception=SimulatedPerception(PerceptionConfig(
                     position_sigma_m=.6, range_sigma_per_m=0,
                     capture_delay_ticks=0, dropout_probability=0), 7),
                 mode="two_stage_viability")
    assert result.perception_planned_steps > 0
    assert seen
    assert all(world is not original for world in seen)
    assert all(world.obstacles != original.obstacles for world in seen)
    assert result.perception_blocked_steps >= 1
    assert result.perception_enabled


def test_oracle_baseline_remains_opt_in_and_unchanged():
    a = run("empty", mode="two_stage_viability",
            config=RunConfig(max_time=.1))
    b = run("empty", mode="two_stage_viability",
            config=RunConfig(max_time=.1), perception=None)
    assert not a.perception_enabled and not b.perception_enabled
    assert [r["command_speed"] for r in a.trace] == [
        r["command_speed"] for r in b.trace]
    assert [r["x"] for r in a.trace] == [r["x"] for r in b.trace]
    assert "perception_source" not in a.trace[0]
