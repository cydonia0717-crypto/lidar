"""Offline independent-visible-target audit for ROS1 LiDAR track exports.

Tracks alone cannot reveal targets that the tracker never produced. A separate
human/independent sensor annotation file marks *visible* objects in the LiDAR
field of view. This code matches contemporaneous positions with a predeclared
geometric gate, marking ambiguous associations as unknown rather than guessing.

Do not call the derived geometric association fraction a calibrated LiDAR
detector recall: annotation visibility, object association and extrinsics have
not been independently proven by software. Never use this tool for actuation.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .tracked_shadow import SnapshotError, parse_snapshot, read_snapshots

VISIBILITIES = frozenset({"visible", "occluded", "out_of_fov", "unknown"})


class VisibilityError(ValueError):
    """Invalid or scientifically ambiguous visibility annotation contract."""


@dataclass(frozen=True)
class Record:
    episode_id: str
    session_id: str
    split: str
    snapshots: Path
    annotations: Path


@dataclass(frozen=True)
class Label:
    time_s: float
    object_id: str
    visibility: str
    x: float | None
    y: float | None


def real(value: str, field: str) -> float:
    try:
        x = float(value)
    except (ValueError, TypeError) as exc:
        raise VisibilityError(f"invalid {field}: {value!r}") from exc
    if not math.isfinite(x):
        raise VisibilityError(f"nonfinite {field}")
    return x


def load_manifest(path: Path) -> list[Record]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"episode_id", "session_id", "split", "snapshot_jsonl",
                    "visibility_csv", "annotation_kind"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise VisibilityError("missing visibility manifest columns")
        entries = []
        for row in reader:
            episode, session, split = (
                row["episode_id"].strip(), row["session_id"].strip(),
                row["split"].strip())
            if not episode or not session or split not in ("calibration", "test"):
                raise VisibilityError("episode_id/session_id/split invalid")
            if row["annotation_kind"] != "independent_visibility":
                raise VisibilityError("annotation_kind must be independent_visibility")
            tracks = (path.parent / row["snapshot_jsonl"]).resolve()
            labels = (path.parent / row["visibility_csv"]).resolve()
            if not tracks.is_file() or not labels.is_file():
                raise VisibilityError(f"missing episode files: {episode}")
            entries.append(Record(episode, session, split, tracks, labels))
    if not entries or len({e.episode_id for e in entries}) != len(entries):
        raise VisibilityError("empty or duplicated episode_id")
    if {e.split for e in entries} != {"calibration", "test"}:
        raise VisibilityError("both calibration and test episodes required")
    for attr in ("snapshots", "annotations"):
        if len({getattr(e, attr) for e in entries}) != len(entries):
            raise VisibilityError(f"source file reused in multiple episodes: {attr}")
    sessions: dict[str, str] = {}
    for e in entries:
        earlier = sessions.setdefault(e.session_id, e.split)
        if earlier != e.split:
            raise VisibilityError(
                f"session leakage across calibration/test: {e.session_id}")
    return entries


def load_visibility(path: Path) -> list[Label]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = {"time_s", "frame", "object_id", "visibility", "x_m", "y_m"}
        if not reader.fieldnames or not fields.issubset(reader.fieldnames):
            raise VisibilityError("visibility CSV missing required columns")
        seen: set[tuple[str, float]] = set()
        output = []
        for lineno, item in enumerate(reader, 2):
            if item["frame"] != "odom":
                raise VisibilityError(f"line {lineno}: expected odom coordinates")
            t = real(item["time_s"], "time_s")
            object_id = item["object_id"].strip()
            kind = item["visibility"]
            if t < 0 or not object_id or kind not in VISIBILITIES:
                raise VisibilityError(f"line {lineno}: invalid visibility label")
            pair = (object_id, t)
            if pair in seen:
                raise VisibilityError(f"line {lineno}: duplicate object/time label")
            seen.add(pair)
            if kind == "visible":
                x = real(item["x_m"], "visible x_m")
                y = real(item["y_m"], "visible y_m")
            else:
                # Position is optional for not-visible/unknown observations;
                # never infer visibility from an arbitrary position.
                x = real(item["x_m"], "x_m") if item["x_m"] else None
                y = real(item["y_m"], "y_m") if item["y_m"] else None
            output.append(Label(t, object_id, kind, x, y))
    if not output:
        raise VisibilityError("independent visibility labels are empty")
    return sorted(output, key=lambda r: (r.time_s, r.object_id))


def load_snapshots(path: Path, max_track_age_s: float):
    records = []
    last_t = -math.inf
    for lineno, raw in read_snapshots(path):
        try:
            parsed = parse_snapshot(raw, max_age_s=max_track_age_s)
        except SnapshotError as exc:
            raise VisibilityError(f"snapshot line {lineno}: {exc}") from exc
        if parsed.time_s <= last_t:
            raise VisibilityError("snapshot time must strictly increase")
        records.append(parsed)
        last_t = parsed.time_s
    if not records:
        raise VisibilityError("no recorded tracker snapshots")
    return records


def nearest_snapshot(times: list[float], t: float, tolerance_s: float) -> int | None:
    idx = bisect.bisect_left(times, t)
    nearest = [j for j in (idx-1, idx) if 0 <= j < len(times)]
    if not nearest:
        return None
    d = min(abs(times[i]-t) for i in nearest)
    if d > tolerance_s:
        return None
    # Two equally close snapshots mean a timestamp association is ambiguous.
    if sum(abs(abs(times[i]-t)-d) <= 1e-9 for i in nearest) > 1:
        return -1
    return min(nearest, key=lambda i: abs(times[i]-t))


def evaluate_record(record: Record, *, gate_m: float = .75,
                    time_tolerance_s: float = .05,
                    max_track_age_s: float = .35) -> list[dict]:
    if not (math.isfinite(gate_m) and gate_m > 0
            and math.isfinite(time_tolerance_s) and time_tolerance_s >= 0):
        raise VisibilityError("invalid gate or clock tolerance")
    stamps = load_visibility(record.annotations)
    snapshots = load_snapshots(record.snapshots, max_track_age_s)
    times = [s.time_s for s in snapshots]
    intermediate = []
    grouped: dict[int, list[tuple[int, set[str]]]] = {}
    for label in stamps:
        base = {
            "episode_id": record.episode_id, "session_id": record.session_id,
            "split": record.split, "object_id": label.object_id,
            "label_t_s": label.time_s, "visibility": label.visibility,
            "matched_snapshot_t_s": "", "associated_track_id": "",
            "geometric_residual_m": "", "status": "",
        }
        if label.visibility != "visible":
            base["status"] = "excluded_not_visible"
            intermediate.append(base)
            continue
        index = nearest_snapshot(times, label.time_s, time_tolerance_s)
        if index is None:
            base["status"] = "no_contemporaneous_tracker_snapshot"
            intermediate.append(base)
            continue
        if index == -1:
            base["status"] = "ambiguous_snapshot_timestamp"
            intermediate.append(base)
            continue
        snapshot = snapshots[index]
        base["matched_snapshot_t_s"] = snapshot.time_s
        candidates: dict[str, float] = {}
        for obstacle in snapshot.world.obstacles:
            if obstacle.name in snapshot.stale_track_ids:
                continue
            ox, oy = obstacle.position(label.time_s)
            residual = math.hypot(label.x-ox, label.y-oy)
            if residual <= gate_m:
                candidates[obstacle.name] = residual
        if not candidates:
            base["status"] = "visible_but_no_gated_track"
        elif len(candidates) > 1:
            base["status"] = "ambiguous_multiple_tracks_in_gate"
        else:
            track_id, error = next(iter(candidates.items()))
            base["status"] = "geometrically_associated"
            base["associated_track_id"] = track_id
            base["geometric_residual_m"] = error
        at = len(intermediate)
        intermediate.append(base)
        grouped.setdefault(index, []).append((at, set(candidates)))
    # Fail closed if one track may represent multiple simultaneously visible
    # labelled objects. Do not silently count duplicate IDs as true positives.
    for _, assignments in grouped.items():
        owners: dict[str, list[int]] = {}
        for index, candidate_ids in assignments:
            for track_id in candidate_ids:
                owners.setdefault(track_id, []).append(index)
        ambiguous = {idx for ids in owners.values() if len(ids) > 1 for idx in ids}
        for idx in ambiguous:
            intermediate[idx]["status"] = "ambiguous_object_track_association"
            intermediate[idx]["associated_track_id"] = ""
            intermediate[idx]["geometric_residual_m"] = ""
    return intermediate


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["split"], row["episode_id"]), []).append(row)
    summary = []
    for (split, episode_id), r in sorted(grouped.items()):
        valid = [item for item in r if item["visibility"] == "visible"]
        counter = Counter(item["status"] for item in valid)
        n = len(valid)
        matched = counter["geometrically_associated"]
        missed = counter["visible_but_no_gated_track"]
        unknown = n - matched - missed
        summary.append({
            "episode_id": episode_id, "split": split,
            "reference_visible_events": n,
            "geometric_associations": matched,
            "visible_but_unmatched": missed,
            "visible_unknown_or_unscorable": unknown,
            "reference_nonvisible_events": len(r)-n,
            "conservative_association_fraction": matched/n if n else "",
            "optimistic_association_fraction": (matched+unknown)/n if n else "",
            "missing_or_ambiguous_denominator_fraction": unknown/n if n else "",
            "interpretation": (
                "geometric_match_only_not_certified_detection_recall"
                if n else "no_independent_visible_targets_scored"
            ),
        })
    return summary


def write_csv(path: Path, values: list[dict]) -> None:
    if not values:
        raise VisibilityError("no audit rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader()
        w.writerows(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent visible-target geometric association audit; no ROS writes")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-m", type=float, default=.75)
    parser.add_argument("--time-tolerance-s", type=float, default=.05)
    parser.add_argument("--max-track-age-s", type=float, default=.35)
    args = parser.parse_args()
    if not (math.isfinite(args.max_track_age_s) and args.max_track_age_s >= 0):
        parser.error("invalid max track age")
    episodes = load_manifest(args.manifest)
    rows = [line for e in episodes for line in evaluate_record(
        e, gate_m=args.gate_m, time_tolerance_s=args.time_tolerance_s,
        max_track_age_s=args.max_track_age_s)]
    summary = summarize(rows)
    write_csv(args.output/"visibility_events.csv", rows)
    write_csv(args.output/"visibility_episode_summary.csv", summary)
    (args.output/"run_config.json").write_text(json.dumps({
        "manifest": str(args.manifest),
        "annotation_kind": "user-declared independent_visibility; not independently verified",
        "gate_m": args.gate_m,
        "time_tolerance_s": args.time_tolerance_s,
        "max_track_age_s": args.max_track_age_s,
        "warning": ("Only contemporaneous geometric association; this is not "
                    "verified detector recall, object ID tracking accuracy, "
                    "prediction coverage or safe navigation."),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in summary:
        print(f"{r['split']}/{r['episode_id']}: visible={r['reference_visible_events']}, "
              f"associated={r['geometric_associations']}, "
              f"unmatched={r['visible_but_unmatched']}, "
              f"unknown={r['visible_unknown_or_unscorable']}")
    print("No ROS messages or actuator commands were published.")


if __name__ == "__main__":
    main()
