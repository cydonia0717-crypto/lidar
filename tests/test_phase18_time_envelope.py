"""Prediction time is not the same thing as obstacle's stale observation age.

All tests use synthetic states, never reference truth as a control signal.
"""
import csv
import math

import pytest

from lidar_sim.evaluation import TrialSpec
from lidar_sim.model import State, VehicleConfig
from lidar_sim.planner import collision_free
from lidar_sim.two_stage import first_violation_kind
from lidar_sim.decision import trajectory_cost_breakdown
from lidar_sim.planner import PlannerConfig
from lidar_sim.phase18_time_envelope_eval import (
    HORIZONS, RAMP_M, POLICIES, calibrate_horizons, study, paired,
)
from lidar_sim.simulated_perception import SimulatedPerception, PerceptionConfig
from lidar_sim.world import MovingObstacle, World


def test_interpolated_radius_is_decision_anchored_not_track_age():
    o = MovingObstacle(
        "estimated", 10, -2, 0, .4, .35,
        reference_time_s=1.1,
        envelope_anchor_s=1.3,
        envelope_horizons_s=(0., .5, 1., 2.),
        envelope_padding_m=(.1, .3, .5, .9))
    assert o.position(1.3) == pytest.approx((10, -1.92))
    assert o.radius == .35
    assert o.radius_at(1.2) == pytest.approx(.45)
    assert o.radius_at(1.3) == pytest.approx(.45)
    assert o.radius_at(1.55) == pytest.approx(.55)
    assert o.radius_at(1.8) == pytest.approx(.65)
    assert o.radius_at(2.8) == pytest.approx(1.05)
    assert o.radius_at(5.) == pytest.approx(1.25)
    assert MovingObstacle("truth", 10, -2, 0, .4).radius_at(20) == .35


@pytest.mark.parametrize("h,v,anchor", [
    ((0., .5), (.1,), 0.),
    ((0., .5), (-.1, .2), 0.),
    ((0., .5), (.1, math.nan), 0.),
    ((.1, .5), (.1, .2), 0.),
    ((0., 0.), (.1, .2), 0.),
    ((0., -.5), (.1, .2), 0.),
    ((0., .5), (.1, .2), None),
    ((), (), 0.),
])
def test_invalid_time_schedule_fails(h,v,anchor):
    with pytest.raises(ValueError, match="envelope"):
        MovingObstacle("bad", 0, 0, 0, 0, .3, 0., anchor, h, v)


def test_hard_check_and_rejection_diagnostic_use_future_time_radius():
    car = VehicleConfig()
    early = State(0, 0, 0, 0, time=1.)
    late = State(0, 0, 0, 0, time=2.)
    cx, cy = car.body_center(early)
    # Planner safety margin is 0.2 m, add 0.2 m physical free distance.
    distance = car.collision_radius + .35 + .2 + .2
    base = MovingObstacle("estimated", cx+distance, cy, 0, 0, .35)
    extra = MovingObstacle("estimated", cx+distance, cy, 0, 0, .35,
                           envelope_anchor_s=1.,
                           envelope_horizons_s=(0.,1.),
                           envelope_padding_m=(0., .3))
    baseline = World((base,))
    estimate = World((extra,))
    assert collision_free([early, late], baseline, car)
    assert collision_free([early], estimate, car)
    assert not collision_free([early, late], estimate, car)
    assert first_violation_kind([early, late], estimate, car, 0.) == "obstacle"
    assert estimate.obstacle_clearance(late, car) < (
        baseline.obstacle_clearance(late, car))
    # A representative hard safety constraint and its explanation must agree.


def test_perception_radius_schedule_never_modifies_physical_truth():
    truth = World((MovingObstacle("person", 10, 0, 0, .3, .35),))
    cfg = PerceptionConfig(position_sigma_m=0, range_sigma_per_m=0,
                           dropout_probability=0, capture_delay_ticks=0)
    sensor = SimulatedPerception(
        cfg, 9, time_padding_horizons_s=HORIZONS,
        time_padding_values_m=RAMP_M)
    # Two actual captures: the speed estimate only becomes available at t=.1
    states = [State(0,0,0,0,time=i*.1) for i in range(2)]
    sensor.observe(truth, states[:1], dt=.1)
    frame = sensor.observe(truth, states, dt=.1)
    assert frame.valid_tracks == 1
    target = frame.estimated_world.obstacles[0]
    assert target.envelope_anchor_s == pytest.approx(.1)
    assert target.reference_time_s == pytest.approx(.1)
    assert target.radius_at(.1) == pytest.approx(.45)
    assert target.radius_at(.6) == pytest.approx(.55)
    assert truth.obstacles[0].radius_at(.6) == .35


def test_schedule_incompatible_with_nonzero_static_padding():
    with pytest.raises(ValueError, match="exclusive"):
        SimulatedPerception(
            PerceptionConfig(), 4, obstacle_padding_m=.3,
            time_padding_horizons_s=HORIZONS,
            time_padding_values_m=RAMP_M)


def test_independent_horizon_calibration_has_true_disjoint_cases():
    specs = [TrialSpec("two_targets", seed, "weak_brake")
             for seed in range(10,19)]
    cal = calibrate_horizons(specs, ticks=24, dt=.1)
    assert len(cal["calibration_case_ids"]) == 9
    for profile, metrics in cal["profiles"].items():
        assert metrics["source_offline_profile"] == profile.removesuffix("_ls5")
        assert metrics["calibration_episode_units"] == 9
        radii = metrics["padding_m"]
        assert len(radii) == 4
        assert radii[0] == radii[1]
        assert radii[1] <= radii[2] <= radii[3]
        assert metrics["joint_coverage_certified"] is False
        assert all(metrics["scored_track_horizon_events"].values())
    with pytest.raises(ValueError, match="unique"):
        calibrate_horizons(specs+specs[:1], ticks=24, dt=.1)


def test_same_case_policies_keep_reference_world_constant(tmp_path):
    cal = {
        "calibration_case_ids": ["two_targets__weak_brake__10"],
        "profiles": {"noisy_ls5": {
            "time_nodes_s": list(HORIZONS),
            "padding_m": [1.,1.1,1.2,1.3]}}}
    case = TrialSpec("two_targets", 60, "weak_brake")
    with pytest.raises(ValueError, match="leaked"):
        study([TrialSpec("two_targets", 10, "weak_brake")],
              calibration=cal, output=tmp_path,
              profiles=("noisy_ls5",))
    results = study([case], calibration=cal, output=tmp_path,
                    profiles=("noisy_ls5",), max_time=.2)
    assert len(results) == len(POLICIES)
    assert len({r["observation_seed"] for r in results}) == 1
    by = {r["policy"]: r for r in results}
    assert by["none"]["extra_radius_at_h1_m"] == 0
    assert by["fixed_030"]["extra_radius_at_h1_m"] == pytest.approx(.3)
    assert by["time_ramp"]["extra_radius_at_h1_m"] == pytest.approx(.3)
    assert by["time_q90"]["extra_radius_at_h1_m"] == pytest.approx(1.2)
    assert all(r["perception_enabled"] for r in results)
    assert len(paired(results)) == len(POLICIES)
    with (tmp_path/"episodes.csv").open(newline="",encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == len(POLICIES)


def test_pair_requires_complete_policy_set():
    with pytest.raises(ValueError, match="incomplete"):
        paired([{"case_id":"one","mode":"two_stage_viability",
                 "sensor_profile":"noisy_ls5","policy":"none",
                 "observation_seed":1}])
