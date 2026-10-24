"""Strictly causal velocity estimator comparison for phase18 synthetic sensing.

All comparisons use the exact same 2D detections and reference truth only in
the *test*, never in the algorithm. The default EMA path stays unchanged.
"""
from dataclasses import replace
import math

import pytest

from lidar_sim.model import State
from lidar_sim.simulated_perception import PerceptionConfig, SimulatedPerception
from lidar_sim.world import World, MovingObstacle


def poses(n, *, base=0.0):
    return [State(0.0, 0.0, 0.0, 0.0, time=base+0.1*i)
            for i in range(n)]


def moving(vx=0.3, vy=0.6, *, base=0):
    # World object's t0 corresponds to the first capture epoch.
    return World((MovingObstacle("target", 11.0, -2.0, vx, vy, 0.4, base),))


def observe_history(observer, world, n, *, base=0.0):
    frames = []
    full = poses(n, base=base)
    for i in range(n):
        frames.append(observer.observe(world, full[:i+1], dt=0.1))
    return frames


def test_ls_exact_cv_from_past_centers_and_no_future_velocity_leak():
    cfg = PerceptionConfig(
        position_sigma_m=0.0, range_sigma_per_m=0.0,
        dropout_probability=0.0, capture_delay_ticks=0,
        velocity_estimator="windowed_ls", regression_window=5)
    observer = SimulatedPerception(cfg, 7)
    world = moving()
    frames = observe_history(observer, world, 9)
    assert frames[0].blocked_reason == "uninitialized_or_no_tracks"
    assert frames[1].valid_tracks == 1
    for frame in frames[1:]:
        target = frame.estimated_world.obstacles[0]
        assert target.vx == pytest.approx(.3, abs=1e-12)
        assert target.vy == pytest.approx(.6, abs=1e-12)
        assert target.position(frame.planning_time_s) == pytest.approx(
            world.obstacles[0].position(frame.planning_time_s), abs=1e-12)


def test_ls_and_ema_have_identical_measurements_but_different_velocities():
    baseline = PerceptionConfig(
        position_sigma_m=.15, range_sigma_per_m=.01,
        dropout_probability=0, capture_delay_ticks=0)
    ls = replace(baseline, velocity_estimator="windowed_ls")
    ema_frames = observe_history(SimulatedPerception(baseline, 31), moving(), 12)
    ls_frames = observe_history(SimulatedPerception(ls, 31), moving(), 12)
    for a, b in zip(ema_frames, ls_frames):
        assert a.observation_count == b.observation_count
        assert a.blocked_reason == b.blocked_reason
        for left, right in zip(a.estimated_world.obstacles,
                               b.estimated_world.obstacles):
            assert left.name == right.name
            assert left.x0 == right.x0
            assert left.y0 == right.y0
            assert left.reference_time_s == right.reference_time_s
    assert any(
        abs(a.estimated_world.obstacles[0].vy -
            b.estimated_world.obstacles[0].vy) > 1e-8
        for a, b in zip(ema_frames[5:], ls_frames[5:]))


def test_ls_replay_is_bitwise_deterministic_even_after_dropouts():
    cfg = PerceptionConfig(
        position_sigma_m=.08, range_sigma_per_m=.005,
        dropout_probability=.25, capture_delay_ticks=1,
        velocity_estimator="windowed_ls", regression_window=4)
    def sample():
        return observe_history(SimulatedPerception(cfg, 91), moving(), 35)
    assert sample() == sample()


def test_ls_uses_only_historic_samples_even_with_epoch_timestamps():
    stamp = 1_792_812_345.0
    cfg = PerceptionConfig(
        position_sigma_m=0, range_sigma_per_m=0,
        dropout_probability=0, capture_delay_ticks=0,
        velocity_estimator="windowed_ls")
    frames = observe_history(
        SimulatedPerception(cfg, 99), moving(base=stamp), 10, base=stamp)
    obs = frames[-1].estimated_world.obstacles[0]
    assert obs.vx == pytest.approx(.3, rel=1e-5)
    assert obs.vy == pytest.approx(.6, rel=1e-5)
    # Never reconstruct an x0 at the zero Unix epoch.


def test_ls_window_two_equals_finite_difference_on_same_detections():
    ema_cfg = PerceptionConfig(
        position_sigma_m=.12, range_sigma_per_m=.002,
        dropout_probability=0, capture_delay_ticks=0,
        velocity_update_gain=1.0)
    ls_cfg = replace(ema_cfg, velocity_estimator="windowed_ls",
                     regression_window=2)
    em = observe_history(SimulatedPerception(ema_cfg, 17), moving(), 13)
    ls = observe_history(SimulatedPerception(ls_cfg, 17), moving(), 13)
    for a, b in zip(em[1:], ls[1:]):
        va = a.estimated_world.obstacles[0]
        vb = b.estimated_world.obstacles[0]
        assert va.vx == pytest.approx(vb.vx, abs=1e-12)
        assert va.vy == pytest.approx(vb.vy, abs=1e-12)


def test_no_noise_even_under_dropped_intermediate_observations():
    cfg = PerceptionConfig(
        position_sigma_m=0, range_sigma_per_m=0,
        dropout_probability=0, capture_delay_ticks=0,
        sensor_every_ticks=3,
        velocity_estimator="windowed_ls", regression_window=5,
        max_track_age_s=.5)
    frames = observe_history(SimulatedPerception(cfg, 7), moving(), 14)
    known = [f for f in frames if f.estimated_world.obstacles]
    assert known
    assert all(f.estimated_world.obstacles[0].vy == pytest.approx(.6)
               for f in known)


@pytest.mark.parametrize("change", [
    {"velocity_estimator": "oracle"},
    {"velocity_estimator": ""},
    {"regression_window": 0},
    {"regression_window": 1},
    {"regression_window": 3.2},
    {"regression_window": True},
])
def test_bad_velocity_estimator_config_rejected(change):
    with pytest.raises(ValueError):
        PerceptionConfig(**change)
