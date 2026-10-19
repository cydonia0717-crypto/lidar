"""Actual episode forensics must preserve physical truth and missing values."""
import math

from lidar_sim.forensics import forensic_report, target_clearances
from lidar_sim.experiment import RunConfig, run
from lidar_sim.model import State, VehicleConfig
from lidar_sim.world import World, MovingObstacle
from lidar_sim.evaluation import TrialSpec, make_trial


def test_signed_obstacle_gap_at_actual_time():
    car = VehicleConfig()
    world = World((MovingObstacle("ped", 1.0, -2.0, 0, 2.0, .2),))
    row = {"time": 1.0, "x": 0, "y": 0}
    result = target_clearances(row, world, car)
    assert result[0][0] == "ped"
    assert math.isclose(result[0][1], 1.0 - car.collision_radius - .2)


def test_forensics_of_guaranteed_initial_collision():
    car = VehicleConfig()
    world = World((MovingObstacle("overlap", 0, 0, 0, 0, 1),))
    result = run("empty", seed=0, mode="braking", vehicle=car,
                 world_override=world, config=RunConfig(max_time=.2))
    report = forensic_report(result, world, car)
    assert report["collision"]
    assert report["impact_target"] == "overlap"
    assert report["impact_signed_gap_m"] < 0
    assert report["recent_trace"][-1]["nearest"] == "overlap"
    assert report["end_time"] == 0


def test_no_target_is_reported_as_none():
    car = VehicleConfig()
    world = World(())
    result = run("empty", seed=0, vehicle=car,
                 world_override=world, config=RunConfig(max_time=.2))
    report = forensic_report(result, world, car)
    assert report["impact_target"] is None
    assert report["impact_signed_gap_m"] is None
    assert report["first_proximity_time"] is None


def test_reproduce_one_two_target_rollout_deterministically():
    trial = make_trial(TrialSpec("two_targets", 0, "nominal"))
    kwargs = dict(vehicle=trial.vehicle, planner=trial.planner,
                  world_override=trial.world, config=RunConfig(max_time=1.0))
    a = run("two_targets", 0, "braking", **kwargs)
    b = run("two_targets", 0, "braking", **kwargs)
    assert forensic_report(a, trial.world, trial.vehicle) == forensic_report(b, trial.world, trial.vehicle)
