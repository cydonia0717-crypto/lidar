"""An oriented physical body may fit the road where a covering disc cannot."""
import csv
from dataclasses import replace
from math import hypot, pi, radians, sin, cos

import pytest

from lidar_sim.model import State, VehicleConfig
from lidar_sim.world import World
from lidar_sim.planner import PlannerConfig, choose_action
from lidar_sim.experiment import RunConfig, run
from lidar_sim.evaluation import (TrialSpec, evaluate_cases,
                                  paired_road_footprint_effects, save_study)
from lidar_sim.road_footprint_audit import (rectangle_half_road_extent,
                                            audit_footprint)


def test_rectangle_is_exact_projection_at_cardinal_headings():
    car = VehicleConfig()
    assert rectangle_half_road_extent(0, car.length, car.width) == pytest.approx(
        car.width / 2)
    assert rectangle_half_road_extent(pi/2, car.length, car.width) == pytest.approx(
        car.length / 2)
    for degrees in (-89, -45, 0, 27.536, 60, 90):
        half=rectangle_half_road_extent(
            radians(degrees), car.length, car.width)
        assert half <= car.collision_radius + 1e-10


def test_same_vehicle_dimensions_and_same_road_safety_margin():
    world=World((), road_half_width=3.25)
    car=VehicleConfig()
    s=State(14.9751,1.964,radians(27.536),0,-.42,20)
    baseline=world.road_clearance(s,car)-world.safety_margin
    precise=world.road_clearance(
        s, replace(car,road_footprint="rectangle"))-world.safety_margin
    assert 0.04 < baseline < .06
    assert .26 < precise < .28
    assert precise > baseline
    # No resizing, new safety margin or more permissive TRUE boundary.
    assert car.width == replace(car,road_footprint="rectangle").width
    assert world.safety_margin == .2


def test_rectangle_does_not_approve_an_actually_outside_car():
    world=World(())
    car=replace(VehicleConfig(),road_footprint="rectangle")
    for yaw in (0, pi/4, pi/2):
        half=rectangle_half_road_extent(yaw,car.length,car.width)
        outside=State(0,world.road_half_width-half+.001,yaw,0)
        assert world.road_clearance(outside,car) < 0


def test_both_planners_keep_equal_target_collision_radius():
    car=VehicleConfig()
    world=World(())
    start=State(0,0,0,1.5)
    orig=choose_action(start,world,car,PlannerConfig(),mode="two_stage")
    new=choose_action(start,world,car,PlannerConfig(),mode="two_stage_rectroad")
    assert orig.backup_checked and new.backup_checked
    assert not orig.unsafe_fallback and not new.unsafe_fallback
    # Extra road room cannot remove a candidate that was feasible with disc.
    assert new.feasible_count >= orig.feasible_count


def test_same_physical_model_across_episodes_and_csv(tmp_path):
    rows=evaluate_cases([TrialSpec("empty",0,"nominal")],
                        ("two_stage","two_stage_rectroad"),
                        max_time=.3,physical_road_footprint="rectangle")
    assert len(rows)==2
    assert {r["physical_road_footprint"] for r in rows}=={"rectangle"}
    assert {r["road_half_width_m"] for r in rows}=={3.25}
    paired=paired_road_footprint_effects(rows)
    assert len(paired)==1
    files=save_study(rows,tmp_path)
    assert files["paired_road_footprint"].exists()
    with files["paired_road_footprint"].open(encoding="utf8",newline="") as fh:
        assert len(list(csv.DictReader(fh)))==1
    with pytest.raises(ValueError,match="common oriented rectangle"):
        paired_road_footprint_effects([
            dict(row, physical_road_footprint="disc") for row in rows])


def test_disallow_a_nonmatching_physical_model_in_paired_effects():
    identical=dict(case_id="two_targets__nominal__0",scenario="two_targets",
                   stress="nominal",seed=0,target_speed_scale=1.0,
                   road_half_width_m=3.25,control_delay_s=.25,braking_mps2=1.6,
                   success=False, collision=False,road_exit=False,
                   timeout=True,mean_planning_ms=13,duration_s=30)
    original=dict(identical,mode="two_stage",physical_road_footprint="rectangle")
    changed=dict(identical,mode="two_stage_rectroad",physical_road_footprint="disc")
    with pytest.raises(ValueError,match="unmatched"):
        paired_road_footprint_effects([original,changed])


def test_diagnostic_does_not_change_run_or_geometry():
    world=World((),road_half_width=3.25)
    baseline=run("empty",mode="two_stage",config=RunConfig(max_time=.2))
    trace=baseline.trace
    before=list(trace)
    a,windows=audit_footprint(trace,3.25,1.9,.85,.2)
    assert len(a)>0 and len(windows)>0
    assert trace==before
    assert "remaining_circle_road_m" in a[0]


def test_invalid_physical_footprint_fails_explicitly():
    with pytest.raises(ValueError,match="unknown"):
        run("empty",config=RunConfig(max_time=.1,physical_road_footprint="smaller"))
    with pytest.raises(ValueError,match="unknown road"):
        World(()).road_clearance(State(0,0,0,0),
                                 replace(VehicleConfig(),road_footprint="fake"))
