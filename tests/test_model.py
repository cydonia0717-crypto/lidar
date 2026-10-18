import pytest

from lidar_sim.model import Command, State, VehicleConfig, step
from lidar_sim.planner import PlannerConfig, choose_action, collision_free, rollout
from lidar_sim.world import MovingObstacle, World


def test_straight_line_and_time():
    cfg = VehicleConfig()
    state = step(State(0, 0, 0, 1.0), Command(1.0, 0), cfg, .1)
    assert state.x == pytest.approx(.1)
    assert state.y == pytest.approx(0)
    assert state.time == pytest.approx(.1)


def test_speed_acceleration_and_steering_limits():
    cfg = VehicleConfig()
    result = step(State(0, 0, 0, 0.0), Command(99, 99), cfg, .1)
    assert result.speed == pytest.approx(cfg.acceleration * .1)
    assert result.steer == pytest.approx(cfg.max_steer_rate * .1)


def test_steering_changes_yaw():
    cfg = VehicleConfig()
    result = step(State(0, 0, 0, 1.0), Command(1.0, 0.3), cfg, 0.1)
    assert result.yaw > 0


def test_time_aligned_prediction_differs_from_frozen_obstacle():
    cfg = VehicleConfig()
    robot = State(0, 0, 0, 1.0)
    path = rollout(robot, Command(1.0, 0.0), cfg, 2.0, .1)
    world = World((MovingObstacle("crossing", 1.0, -2.0, 0.0, 2.0, .1),), road_half_width=3.5)
    assert not collision_free(path, world, cfg, predict=True)
    assert collision_free(path, world, cfg, predict=False)


def test_emergency_when_road_too_narrow():
    cfg = VehicleConfig()
    result = choose_action(State(0, 0, 0, 1), World((), road_half_width=.5), cfg, PlannerConfig())
    assert result.emergency and result.feasible_count == 0
    assert result.command.target_speed == 0.0


def test_non_positive_dt_fails():
    with pytest.raises(ValueError):
        step(State(0, 0, 0, 1), Command(1, 0), VehicleConfig(), 0)
