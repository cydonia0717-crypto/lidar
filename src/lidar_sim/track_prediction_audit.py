"""Offline H1 audit: estimated LiDAR tracks vs INDEPENDENT reference labels.

No ROS dependency, no plant actuation, no use of synthetic world ground truth
as planner inputs. Reference labels are used only after prediction to score
future positions. For every episode/group use the *worst complete forecast*
as one calibration/test unit to avoid treating correlated 10 Hz frames as
independent trials. Empirical coverage is not a vehicle collision guarantee.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .tracked_shadow import SnapshotError, parse_snapshot, read_snapshots


class AuditError(ValueError):
    """Invalid data split, time series or independent-reference contract."""


@dataclass(frozen=True)
class Episode:
    episode_id: str
    split: str
    snapshots: Path
    labels: Path


def _number(raw: str, field: str) -> float:
    try:
        x = float(raw)
    except (ValueError, TypeError) as exc:
        raise AuditError(f"invalid {field}: {raw!r}") from exc
    if not math.isfinite(x):
        raise AuditError(f"nonfinite {field}")
    return x


def load_manifest(manifest: Path) -> list[Episode]:
    episodes = []
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"episode_id", "split", "snapshot_jsonl",
                    "reference_csv", "reference_kind"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise AuditError(f"manifest missing columns: {sorted(required)}")
        for r in reader:
            episode_id = r["episode_id"].strip()
            split = r["split"].strip()
            if not episode_id or split not in ("calibration", "test"):
                raise AuditError("nonempty episode_id and calibration/test split required")
            if r["reference_kind"] != "independent_annotation":
                raise AuditError("reference_kind must explicitly be independent_annotation")
            if any(e.episode_id == episode_id for e in episodes):
                raise AuditError(f"duplicate episode_id across splits: {episode_id}")
            snapshots = (manifest.parent / r["snapshot_jsonl"]).resolve()
            labels = (manifest.parent / r["reference_csv"]).resolve()
            if not snapshots.is_file() or not labels.is_file():
                raise AuditError(f"missing files for episode: {episode_id}")
            episodes.append(Episode(episode_id, split, snapshots, labels))
    if not any(e.split == "calibration" for e in episodes) or not any(
            e.split == "test" for e in episodes):
        raise AuditError("both independent calibration and test episodes required")
    # Path-level leakage is a strong warning even when IDs have been renamed.
    for attr in ("snapshots", "labels"):
        paths = [getattr(e, attr) for e in episodes]
        if len(paths) != len(set(paths)):
            raise AuditError(f"shared source file across episodes: {attr}")
    return episodes


def load_labels(path: Path) -> dict[str, tuple[list[float], list[tuple[float, float]]]]:
    """Exact independent reference at observed timestamps; no interpolation."""
    grouped = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"time_s", "frame", "track_id", "x_m", "y_m"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise AuditError("reference CSV missing time/frame/id/position columns")
        for line, row in enumerate(reader, 2):
            if row["frame"] != "odom":
                raise AuditError(f"reference line {line}: frame must be odom")
            track_id = row["track_id"]
            if not track_id:
                raise AuditError(f"reference line {line}: missing track_id")
            t = _number(row["time_s"], "reference time_s")
            x, y = _number(row["x_m"], "reference x_m"), _number(row["y_m"], "reference y_m")
            if t < 0:
                raise AuditError("negative reference timestamp")
            grouped[track_id].append((t, (x, y)))
    result = {}
    for key, records in grouped.items():
        ordered = sorted(records)
        times = [item[0] for item in ordered]
        if any(a == b for a, b in zip(times, times[1:])):
            raise AuditError(f"duplicated reference timestamp for {key}")
        result[key] = (times, [item[1] for item in ordered])
    if not result:
        raise AuditError("empty independent reference labels")
    return result


def reference_at(labels: dict, track_id: str, target: float,
                 tolerance: float) -> tuple[float, float] | None:
    if track_id not in labels:
        return None
    times, poses = labels[track_id]
    index = bisect.bisect_left(times, target)
    candidates = [i for i in (index - 1, index) if 0 <= i < len(times)]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(times[i] - target))
    if abs(times[best] - target) > tolerance:
        return None
    # Equidistant labels imply ambiguous temporal matching; do not guess.
    if (len(candidates) == 2
            and abs(abs(times[candidates[0]] - target)
                    - abs(times[candidates[1]] - target)) <= 1e-9):
        raise AuditError(f"ambiguous time match for {track_id} at {target}")
    return poses[best]


def quality_group(track: Mapping[str, Any], now_s: float, *,
                  good_points: int, max_good_age_s: float) -> str:
    count = track.get("point_count", -1)
    age = now_s - float(track["observed_at_s"])
    return ("good_observation" if (isinstance(count, int)
            and not isinstance(count, bool) and count >= good_points
            and age <= max_good_age_s) else "degraded_or_unknown")


def evaluate_episode(episode: Episode, horizons: tuple[float, ...],
                     tolerance_s: float, *,
                     good_points: int, max_good_age_s: float,
                     max_track_age_s: float,
                     horizon_rows: list[dict] | None = None) -> list[dict]:
    labels = load_labels(episode.labels)
    stats = defaultdict(lambda: {"origins": 0, "complete": 0,
                                 "missing": 0, "worst": -math.inf})
    last_time = -math.inf
    for line_no, item in read_snapshots(episode.snapshots):
        try:
            snapshot = parse_snapshot(item, max_age_s=max_track_age_s)
        except SnapshotError as exc:
            raise AuditError(f"{episode.episode_id} line {line_no}: {exc}") from exc
        if snapshot.time_s <= last_time:
            raise AuditError(f"{episode.episode_id}: non-increasing snapshot times")
        last_time = snapshot.time_s
        for track, obstacle in zip(item["tracks"], snapshot.world.obstacles):
            group = quality_group(track, snapshot.time_s,
                                  good_points=good_points,
                                  max_good_age_s=max_good_age_s)
            entry = stats[group]
            entry["origins"] += 1
            stale = track["id"] in snapshot.stale_track_ids
            errors = []
            for h in horizons:
                t = snapshot.time_s + h
                # A stale observation is not eligible for prediction-quality
                # claims, even if an external reference happens to exist.
                truth = (None if stale else
                         reference_at(labels, track["id"], t, tolerance_s))
                px, py = obstacle.position(t)
                error = (math.hypot(px - truth[0], py - truth[1])
                         if truth is not None else None)
                if horizon_rows is not None:
                    horizon_rows.append({
                        "episode_id": episode.episode_id,
                        "split": episode.split,
                        "track_id": track["id"],
                        "quality_group": group,
                        "origin_t_s": snapshot.time_s,
                        "observed_at_s": track["observed_at_s"],
                        "observation_age_s": snapshot.time_s-track["observed_at_s"],
                        "horizon_s": h,
                        "reference_target_t_s": t,
                        "prediction_x_m": px,
                        "prediction_y_m": py,
                        "reference_x_m": (truth[0] if truth is not None else ""),
                        "reference_y_m": (truth[1] if truth is not None else ""),
                        "radial_error_m": (error if error is not None else ""),
                        "status": ("stale_track" if stale else
                                   "missing_independent_reference"
                                   if truth is None else "matched"),
                    })
                if error is not None:
                    errors.append(error)
            if len(errors) != len(horizons):
                # A missing/stale horizon invalidates the complete segment.
                entry["missing"] += 1
            else:
                entry["complete"] += 1
                entry["worst"] = max(entry["worst"], max(errors))
    rows = []
    for group, values in sorted(stats.items()):
        rows.append({
            "episode_id": episode.episode_id,
            "split": episode.split,
            "quality_group": group,
            "prediction_origins": values["origins"],
            "complete_segments": values["complete"],
            "incomplete_or_stale_segments": values["missing"],
            "worst_complete_segment_error_m": (
                values["worst"] if values["complete"] else ""),
        })
    if not rows:
        rows.append({
            "episode_id": episode.episode_id, "split": episode.split,
            "quality_group": "no_observed_tracks",
            "prediction_origins": 0, "complete_segments": 0,
            "incomplete_or_stale_segments": 0,
            "worst_complete_segment_error_m": "",
        })
    return rows


def conformal_quantile(values: list[float], alpha: float) -> float | None:
    """Episode-level split conformal rank; None means insufficient units.

    Its marginal guarantee needs exchangeable independent episode/group
    observations and valid reference labels. We do not claim those conditions
    from a JSON file or automatically provide a formal safety guarantee.
    """
    if not 0 < alpha < 1:
        raise AuditError("alpha must be between 0 and 1")
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise AuditError("calibration errors must be nonnegative and finite")
    n = len(values)
    k = math.ceil((n+1)*(1-alpha))
    if k > n:
        return None
    return sorted(values)[k-1]


def _fully_scored_group(row: Mapping[str, Any]) -> bool:
    """A group episode is usable only when *every* prediction origin is scored.

    Complete-case selection over partially annotated/occluded episodes gives a
    systematically optimistic empirical coverage, even if we report missing
    counts elsewhere. The unit is an episode/group, never an individual frame.
    """
    total = int(row["prediction_origins"])
    complete = int(row["complete_segments"])
    missing = int(row["incomplete_or_stale_segments"])
    if min(total, complete, missing) < 0 or complete + missing != total:
        raise AuditError("inconsistent episode/group segment counts")
    return (total > 0 and missing == 0 and complete == total
            and row["worst_complete_segment_error_m"] != "")


def summarize(rows: list[dict], alpha: float) -> list[dict]:
    """Show conditional complete-case coverage AND missing-as-failure bounds.

    A test episode/group containing even one missing forecast remains in the
    denominator of the lower-bound coverage. 'Unscored' cannot become 'safe'.
    Pooled calibration requires every observed group in that episode complete;
    grouped calibration requires the respective group completely scored.
    Neither construction provides a guarantee under informative missingness.
    """
    calibrations: dict[str, list[dict]] = defaultdict(list)
    heldout = [r for r in rows if r["split"] == "test"]
    for r in rows:
        if r["split"] == "calibration":
            calibrations[r["episode_id"]].append(r)
    complete_pooled = {}
    for episode_id, episode_rows in calibrations.items():
        if episode_rows and all(_fully_scored_group(r) for r in episode_rows):
            complete_pooled[episode_id] = max(
                float(r["worst_complete_segment_error_m"])
                for r in episode_rows)
    pooled_q = conformal_quantile(list(complete_pooled.values()), alpha)
    groups = sorted({r["quality_group"] for r in rows if
                     r["quality_group"] != "no_observed_tracks"})
    missing_whole_episodes = sum(
        r["quality_group"] == "no_observed_tracks" for r in heldout)
    results = []
    for group in groups:
        cal = [r for r in rows if r["split"] == "calibration"
               and r["quality_group"] == group]
        cal_values = [float(r["worst_complete_segment_error_m"])
                      for r in cal if _fully_scored_group(r)]
        q = conformal_quantile(cal_values, alpha)
        trials = [r for r in heldout if r["quality_group"] == group]
        complete = [r for r in trials if _fully_scored_group(r)]
        total = len(trials)
        missing = total - len(complete)
        for method, radius in (("pooled", pooled_q), ("grouped", q)):
            success = (sum(float(r["worst_complete_segment_error_m"]) <= radius
                           for r in complete) if radius is not None else None)
            observed = success / len(complete) if (
                success is not None and complete) else ""
            lower = success / total if (
                success is not None and total) else ""
            upper = (success + missing) / total if (
                success is not None and total) else ""
            results.append({
                "quality_group": group, "method": method,
                "alpha_target": alpha,
                "calibration_episode_units": (
                    len(complete_pooled) if method == "pooled" else len(cal_values)),
                "calibration_incomplete_group_units": sum(
                    not _fully_scored_group(r) for r in cal),
                "test_episode_units": total,
                "test_complete_episode_units": len(complete),
                "test_unscored_episode_units": missing,
                # A dataset with zero observed tracks cannot be allocated to a
                # quality group; expose this excluded denominator explicitly.
                "test_whole_episode_no_tracks_units": missing_whole_episodes,
                "radius_m": radius if radius is not None else "",
                "observed_covered_complete": (
                    success if success is not None else ""),
                "observed_fraction": observed,
                "missing_as_failure_lower_bound": lower,
                "missing_as_success_upper_bound": upper,
                "status": ("insufficient_independent_calibration_episodes"
                           if radius is None else
                           "empirical_coverage_not_collision_probability"),
            })
    return results

def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise AuditError("no segment rows to write")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Offline independent-label track prediction error audit")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--horizons", default="0.5,1.0,2.0")
    p.add_argument("--reference-tolerance-s", type=float, default=.05)
    p.add_argument("--alpha", type=float, default=.10)
    p.add_argument("--good-min-points", type=int, default=10)
    p.add_argument("--good-max-age-s", type=float, default=.15)
    p.add_argument("--max-track-age-s", type=float, default=.35)
    a = p.parse_args()
    horizons = tuple(float(x) for x in a.horizons.split(","))
    if not horizons or any(not math.isfinite(h) or h <= 0 for h in horizons):
        p.error("positive finite prediction horizons required")
    if sorted(set(horizons)) != list(horizons):
        p.error("prediction horizons must be unique and sorted")
    if (not math.isfinite(a.reference_tolerance_s)
            or a.reference_tolerance_s < 0
            or not 0 < a.alpha < 1
            or a.good_min_points < 0
            or not math.isfinite(a.good_max_age_s)
            or a.good_max_age_s < 0
            or not math.isfinite(a.max_track_age_s)
            or a.max_track_age_s < 0):
        p.error("invalid matching/quality configuration")
    episodes = load_manifest(a.manifest)
    horizon_rows: list[dict] = []
    rows = [row for e in episodes for row in evaluate_episode(
        e, horizons, a.reference_tolerance_s,
        good_points=a.good_min_points,
        max_good_age_s=a.good_max_age_s,
        max_track_age_s=a.max_track_age_s, horizon_rows=horizon_rows)]
    summary = summarize(rows, a.alpha)
    write_rows(a.output / "episode_segment_error.csv", rows)
    if horizon_rows:
        write_rows(a.output / "per_horizon_residuals.csv", horizon_rows)
    if summary:
        write_rows(a.output / "coverage_audit.csv", summary)
    (a.output / "run_config.json").write_text(json.dumps({
        "manifest": str(a.manifest),
        "reference_kind": "user-declared independent_annotation; not verified",
        "horizons_s": horizons,
        "match_tolerance_s": a.reference_tolerance_s,
        "alpha": a.alpha,
        "good_min_points": a.good_min_points,
        "good_max_age_s": a.good_max_age_s,
        "max_track_age_s": a.max_track_age_s,
        "calibration_episodes": sum(e.split == "calibration" for e in episodes),
        "test_episodes": sum(e.split == "test" for e in episodes),
        "note": "empirical segment coverage, not LiDAR-certified or collision guarantee",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in summary:
        print(f"{r['quality_group']} {r['method']}: radius={r['radius_m']} "
              f"calibration episode units={r['calibration_episode_units']}, "
              f"test complete units={r['test_complete_episode_units']}, "
              f"coverage={r['observed_fraction']}, status={r['status']}")
    print("Reference data is never used to form planner predictions; all output is offline.")


if __name__ == "__main__":
    main()
