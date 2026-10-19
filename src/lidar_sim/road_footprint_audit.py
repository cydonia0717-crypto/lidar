"""Physical road-footprint audit of the proven two-target deadlock.

No changes to the real controller or its safety checks. A rectangle here is
an analytic counterfactual to the conservative centered vehicle-covering disc.
Both use the identical length and width; this is not resizing the vehicle.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
from statistics import mean

from .evaluation import TrialSpec, make_trial
from .experiment import run, RunConfig


def rectangle_half_road_extent(yaw: float, length: float, width: float) -> float:
    """Exact y-axis half projection of a centered yaw-rotated rectangle."""
    return .5 * (abs(length * math.sin(yaw)) +
                 abs(width * math.cos(yaw)))


def audit_footprint(trace: list[dict], road_half_width: float,
                    length: float, width: float, margin: float,
                    *, window_s: float = 5.0) -> tuple[list[dict], list[dict]]:
    if not trace:
        raise ValueError("empty trace")
    if window_s <= 0 or length <= 0 or width <= 0:
        raise ValueError("invalid physical dimensions or window")
    radius = math.hypot(length/2, width/2)
    rows=[]
    for r in trace:
        if not isinstance(r.get("command_speed"), (int, float)):
            continue
        yaw=float(r["yaw"]); y=float(r["y"])
        half=rectangle_half_road_extent(yaw,length,width)
        circle=road_half_width-abs(y)-radius-margin
        rect=road_half_width-abs(y)-half-margin
        row={"time":round(float(r["time"]),4),
             "x":round(float(r["x"]),4), "y":round(y,4),
             "yaw_rad":round(yaw,4), "yaw_deg":round(math.degrees(yaw),3),
             "steer_rad":round(float(r["steer"]),4),
             "speed":round(float(r["speed"]),4),
             "command_speed":r["command_speed"],
             "command_steer":r["command_steer"],
             "road_half_width_m":road_half_width,
             "half_extent_circle_m":round(radius,4),
             "half_extent_rectangle_m":round(half,4),
             "remaining_circle_road_m":round(circle,4),
             "remaining_rectangle_road_m":round(rect,4),
             "moving_first_action_feasible":int(r.get("moving_first_action_feasible",0)),
             "feasible_pairs":int(r.get("feasible_pairs",0)),
             "second_road_rejections":int(r.get("reject_second_road",0))}
        rows.append(row)
    if not rows:
        raise ValueError("no control frames")
    windows=[]
    t0=rows[0]["time"]
    for k in range(int(math.ceil((rows[-1]["time"]-t0+1e-5)/window_s))):
        t=t0+k*window_s
        rs=[r for r in rows if t<=r["time"]<t+window_s]
        if not rs:continue
        windows.append({
            "from_s":t,"to_s":t+window_s,
            "start_x":rs[0]["x"],"end_x":rs[-1]["x"],
            "start_y":rs[0]["y"],"end_y":rs[-1]["y"],
            "mean_yaw_deg":round(mean(r["yaw_deg"] for r in rs),3),
            "end_yaw_deg":rs[-1]["yaw_deg"],
            "end_steer_rad":rs[-1]["steer_rad"],
            "min_circle_road_m":min(r["remaining_circle_road_m"] for r in rs),
            "min_rectangle_road_m":min(r["remaining_rectangle_road_m"] for r in rs),
            "mean_feasible_pairs":round(mean(r["feasible_pairs"] for r in rs),3),
            "forward_feasible_fraction":round(mean(r["moving_first_action_feasible"] for r in rs),3),
            "second_road_rejections":sum(r["second_road_rejections"] for r in rs),
        })
    return rows,windows


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--stress",default="nominal")
    ap.add_argument("--output",default="results/road-footprint-audit")
    args=ap.parse_args(argv)
    trial=make_trial(TrialSpec("two_targets",args.seed,args.stress))
    outcome=run("two_targets",args.seed,"two_stage",vehicle=trial.vehicle,
                planner=trial.planner,world_override=trial.world,
                config=RunConfig(max_time=30))
    rows,windows=audit_footprint(
        outcome.trace,trial.world.road_half_width,
        trial.vehicle.length,trial.vehicle.width,trial.world.safety_margin)
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    for name,data in (("frames.csv",rows),("windows.csv",windows)):
        with (folder/name).open("w",newline="",encoding="utf8") as file:
            wr=csv.DictWriter(file,fieldnames=list(data[0]));wr.writeheader();wr.writerows(data)
    payload={"success":outcome.success,"collision":outcome.collision,
             "timeout":outcome.timeout,"seed":args.seed,
             "stress":args.stress,"windows":windows}
    (folder/"summary.json").write_text(json.dumps(payload,indent=2),encoding="utf8")
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__":
    main()
