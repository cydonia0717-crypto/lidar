"""Fail-closed exact closed-loop equivalence audit, full trajectories required.

The fast planner is an acceleration of the exhaustive *same* viable-first
policy, not a new safety model or an alternative objective. Compare every
executed command and dynamic state under identical fixed episode worlds.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path


FIELDS=("time","x","y","yaw","speed","steer",
        "command_speed","command_steer","applied_speed","applied_steer",
        "body_center_x","body_center_y","pose_gate_met")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="",encoding="utf8") as f:
        return list(csv.DictReader(f))


def equal_value(a: str | None,b: str | None, *, abs_tol: float=1e-8)->bool:
    if a==b:
        return True
    if a is None or b is None:
        return False
    try:
        x=float(a); y=float(b)
    except ValueError:
        return False
    return math.isfinite(x) and math.isfinite(y) and abs(x-y) <= abs_tol


def audit_pair(base: list[dict], optimized: list[dict], *, abs_tol: float=1e-8) -> dict:
    if not base or not optimized:
        raise ValueError("cannot certify empty episode traces")
    n=min(len(base),len(optimized))
    for i in range(n):
        for key in FIELDS:
            if not equal_value(base[i].get(key),optimized[i].get(key),abs_tol=abs_tol):
                return {"equivalent":False,"base_frames":len(base),
                        "fast_frames":len(optimized),"first_difference_frame":i,
                        "first_difference_field":key,
                        "old_value":base[i].get(key),"fast_value":optimized[i].get(key)}
    return {"equivalent":len(base)==len(optimized),
            "base_frames":len(base),"fast_frames":len(optimized),
            "first_difference_frame":None if len(base)==len(optimized) else n,
            "first_difference_field":None if len(base)==len(optimized) else "frame_count"}


def audit_folder(folder: Path)->list[dict]:
    traces=folder/"all_traces"
    if not traces.is_dir():
        raise FileNotFoundError("missing complete per-frame traces: "+str(traces))
    cases={}
    for path in sorted(traces.glob("*.csv")):
        stem=path.stem
        for mode in ("two_stage_viability_fast","two_stage_viability"):
            suffix="__"+mode
            if stem.endswith(suffix):
                key=stem[:-len(suffix)]
                cases.setdefault(key,{})[mode]=path
                break
    if not cases:
        raise ValueError("no paired exhaustive-vs-fast traces")
    rows=[]
    for key,pair in sorted(cases.items()):
        expected={"two_stage_viability_fast","two_stage_viability"}
        if set(pair)!=expected:
            raise ValueError("missing mode for paired case "+key)
        data=audit_pair(read_csv(pair["two_stage_viability"]),
                        read_csv(pair["two_stage_viability_fast"]))
        rows.append({"case_id":key,**data})
    return rows


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--folders",nargs="+",required=True)
    ap.add_argument("--output",default="results/phase16-equivalence.csv")
    args=ap.parse_args(argv)
    rows=[]
    for name in args.folders:
        rows.extend(audit_folder(Path(name)))
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(
            key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows,indent=2),flush=True)
    if not all(row["equivalent"] for row in rows):
        raise AssertionError("fast mode changed executed control or plant trajectory")


if __name__=="__main__":
    main()
