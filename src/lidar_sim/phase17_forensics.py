"""Forensic comparison of phase17 no-feedback vs CPU-lagged closed loops.

Extracts *observed* timing and physical divergence. This tool does not claim
that the first divergent command individually caused the later collision.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def first_difference(a: list[dict], b: list[dict], fields: tuple[str, ...],
                     tol: float = 1e-8) -> float | None:
    for row_a, row_b in zip(a, b):
        for key in fields:
            x, y = row_a.get(key, ""), row_b.get(key, "")
            if x == y:
                continue
            if x in ("", None) or y in ("", None):
                return float(row_a["time"])
            try:
                different = not math.isclose(float(x), float(y), abs_tol=tol)
            except ValueError:
                different = x != y
            if different:
                return float(row_a["time"])
    if len(a) != len(b):
        return float((a if len(a) > len(b) else b)[min(len(a), len(b))]["time"])
    return None


def first_event(rows: list[dict], key: str, expected: str = "1") -> float | None:
    return next((float(r["time"]) for r in rows
                 if r.get(key) == expected), None)


def final_parked_since(rows: list[dict], tolerance: float = 1e-7) -> float | None:
    """Earliest point in the final continuously stopped, position-fixed tail."""
    final = rows[-1]
    if abs(float(final["speed"])) > tolerance:
        return None
    x, y = float(final["x"]), float(final["y"])
    beginning = float(final["time"])
    for row in reversed(rows[:-1]):
        if (abs(float(row["speed"])) > tolerance
                or math.hypot(float(row["x"]) - x,
                              float(row["y"]) - y) > tolerance):
            break
        beginning = float(row["time"])
    return beginning


def audit_case(none_path: Path, measured_path: Path) -> dict:
    base, delayed = load(none_path), load(measured_path)
    if not base or not delayed:
        raise ValueError("empty decision trace")
    body = measured_path.parent.parent.parent
    mode = measured_path.parent.parent.name
    metric_path = measured_path.with_name(measured_path.name.replace(
        "_trace.csv", "_metrics.json"))
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    halted = final_parked_since(delayed)
    final_t = float(delayed[-1]["time"])
    collided_stationary = bool(
        metrics["collision"] and halted is not None
        and final_t - halted >= 2.0)
    return {
        "case_id": body.name,
        "mode": mode,
        "none_terminal_time": float(base[-1]["time"]),
        "delayed_terminal_time": final_t,
        "first_issued_command_divergence_s": first_difference(
            base, delayed, ("command_speed", "command_steer")),
        "first_body_state_divergence_s": first_difference(
            base, delayed, ("x", "y", "yaw", "speed", "steer")),
        "first_busy_tick_s": first_event(delayed, "compute_busy"),
        "first_unsafe_fallback_s": first_event(delayed, "unsafe_fallback"),
        "parked_since_s": halted,
        "stationary_contact_after_2s": collided_stationary,
        "terminal_speed_mps": float(delayed[-1]["speed"]),
        "terminal_x_m": float(delayed[-1]["x"]),
        "terminal_y_m": float(delayed[-1]["y"]),
        "terminal_clearance_m": float(delayed[-1]["min_clearance"]),
        "collision": bool(metrics["collision"]),
        "road_exit": bool(metrics["road_exit"]),
        "success": bool(metrics["success"]),
        "missed_control_ticks": int(metrics["missed_control_ticks"]),
        "unsafe_fallback_steps": int(metrics["unsafe_fallback_steps"]),
    }


def audit_root(root: Path) -> list[dict]:
    reports = []
    for measured in sorted(root.glob("*/*/measured/*_trace.csv")):
        baseline = measured.parent.parent / "none" / measured.name
        if not baseline.exists():
            raise ValueError(f"missing matched no-delay trace: {baseline}")
        reports.append(audit_case(baseline, measured))
    if not reports:
        raise ValueError(f"no measured/none pairs in {root}")
    return reports


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--output", type=Path,
                   default=Path("results/phase17-forensics.csv"))
    args = p.parse_args()
    rows = [r for root in args.roots for r in audit_root(root)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        if r["collision"] or r["stationary_contact_after_2s"]:
            print(f"{r['case_id']}/{r['mode']}: collision={r['collision']} "
                  f"first_no_backup={r['first_unsafe_fallback_s']}s "
                  f"parked_since={r['parked_since_s']}s "
                  f"impact={r['delayed_terminal_time']}s", flush=True)
    print(f"Audited {len(rows)} paired time-series.")


if __name__ == "__main__":
    main()
