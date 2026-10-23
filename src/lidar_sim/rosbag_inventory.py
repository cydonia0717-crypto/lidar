"""Read-only inventory of ROS1 Noetic bags before exporting LiDAR tracks.

Uses ROS1 'rosbag' ONLY when invoked. The research simulator and GitHub CI do
not import ROS; this module can be unit-tested with a synthetic bag facade.

This is an INTEGRITY INVENTORY, not a detector, localization system, TF
transform, track estimator, ground truth, or sensor-calibration pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable


def _secs(stamp: Any) -> float | None:
    """Recognize rospy.Time-like objects and plain numeric test timestamps."""
    if stamp is None:
        return None
    try:
        value = (float(stamp.to_sec()) if callable(getattr(stamp, "to_sec", None))
                 else float(stamp))
    except (ValueError, TypeError, AttributeError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def inventory_bag(
    path: Path, *, sample_per_topic: int = 25,
    bag_factory: Callable[..., Any] | None = None,
) -> tuple[list[dict], dict]:
    """Inventory ROS topic metadata, finite header stamps and sample skew.

    Bag timestamps refer to recording reception; header stamps refer to
    acquisition/processing as assigned upstream. This is not clock sync.
    Only the first N messages per topic are sampled; no global statistics are
    claimed for an entire long recording.
    """
    if sample_per_topic < 1:
        raise ValueError("sample_per_topic must be positive")
    if not path.is_file():
        raise FileNotFoundError(path)
    if bag_factory is None:
        try:
            import rosbag  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "ROS1 rosbag is unavailable: run this on an Ubuntu 20.04 "
                "ROS Noetic system with ROS sourced, not on generic CI"
            ) from exc
        bag_factory = rosbag.Bag
    rows = []
    with bag_factory(str(path), "r") as bag:
        info = bag.get_type_and_topic_info()
        for topic, meta in sorted(info.topics.items()):
            frames = set()
            skews = []
            first_recorded = last_recorded = None
            samples = stamped = 0
            for _, msg, bag_stamp in bag.read_messages(topics=[topic]):
                if samples >= sample_per_topic:
                    break
                samples += 1
                bag_time = _secs(bag_stamp)
                if bag_time is not None:
                    if first_recorded is None:
                        first_recorded = bag_time
                    last_recorded = bag_time
                header = getattr(msg, "header", None)
                stamp = _secs(getattr(header, "stamp", None))
                frame = getattr(header, "frame_id", None)
                if isinstance(frame, str) and frame:
                    frames.add(frame)
                if stamp is not None:
                    stamped += 1
                    if bag_time is not None:
                        skews.append(bag_time - stamp)
            mean_skew = sum(skews) / len(skews) if skews else None
            rows.append({
                "topic": topic,
                "ros_type": str(meta.msg_type),
                "recorded_message_count": int(meta.message_count),
                "nominal_frequency_hz": (
                    float(meta.frequency) if meta.frequency is not None else ""),
                "sampled_messages": samples,
                "sampled_with_valid_header_stamp": stamped,
                "sampled_without_valid_header_stamp": samples-stamped,
                "sampled_frame_ids": "|".join(sorted(frames)),
                "multiple_sampled_frames": int(len(frames) > 1),
                "sampled_mean_record_minus_header_s": (
                    mean_skew if mean_skew is not None else ""),
                "sampled_first_bag_time_s": (
                    first_recorded if first_recorded is not None else ""),
                "sampled_last_bag_time_s": (
                    last_recorded if last_recorded is not None else ""),
                "status": (
                    "no_sampled_messages" if samples == 0 else
                    "mixed_header_frames_needs_investigation" if len(frames) > 1
                    else "some_missing_header_stamps" if stamped < samples else
                    "header_stamps_observed_not_validated"),
            })
        start = bag.get_start_time()
        end = bag.get_end_time()
    # Do not claim that sampled valid stamps imply synchronized sensors,
    # odometry accuracy, tracking readiness, independent ground truth, or TF.
    summary = {
        "source": str(path), "read_only": True,
        "ros_version": "ROS1",
        "sample_per_topic": sample_per_topic,
        "bag_start_s": start, "bag_end_s": end,
        "topic_count": len(rows),
        "topics_with_sampled_valid_stamps": sum(
            r["sampled_with_valid_header_stamp"] > 0 for r in rows),
        "topics_with_multiple_sampled_frames": sum(
            r["multiple_sampled_frames"] for r in rows),
        "warning": (
            "Header frames/stamps are metadata only. Inspect TF and "
            "actuator queue history; no data were transformed or exported "
            "as planning snapshots. Never use KF future output as "
            "independent ground truth."
        ),
    }
    return rows, summary


def write_inventory(rows: list[dict], summary: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no topics; empty ROS bag")
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(
        summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only ROS1 bag topic/time/frame inventory")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-per-topic", type=int, default=25)
    args = parser.parse_args()
    rows, summary = inventory_bag(
        args.bag, sample_per_topic=args.sample_per_topic)
    write_inventory(rows, summary, args.output)
    print(f"Inspected {len(rows)} topics; no messages modified or published.")
    print(f"Topic inventory: {args.output}")
    print("This inventory does not identify verified odom transforms, "
          "physical command latency or independent truth labels.")


if __name__ == "__main__":
    main()
