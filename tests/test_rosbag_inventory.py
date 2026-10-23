"""ROS1 bag reader is optional and inventories metadata without ROS in CI."""
import csv
import json
from types import SimpleNamespace

import pytest

from lidar_sim.rosbag_inventory import inventory_bag, write_inventory


class Stamp:
    def __init__(self, value):
        self.value = value

    def to_sec(self):
        return self.value


class FakeBag:
    def __init__(self, path, mode):
        assert mode == "r"
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get_type_and_topic_info(self):
        return SimpleNamespace(topics={
            "/lidar/points": SimpleNamespace(
                msg_type="sensor_msgs/PointCloud2",
                message_count=3, frequency=10.0),
            "/tracking": SimpleNamespace(
                msg_type="geometry_msgs/PoseArray",
                message_count=2, frequency=None),
        })

    def read_messages(self, *, topics):
        topic = topics[0]
        if topic == "/lidar/points":
            yield (topic, SimpleNamespace(
                header=SimpleNamespace(stamp=Stamp(12.0),
                                       frame_id="lidar")), Stamp(12.1))
            yield (topic, SimpleNamespace(
                header=SimpleNamespace(stamp=Stamp(12.1),
                                       frame_id="lidar")), Stamp(12.2))
            yield (topic, SimpleNamespace(
                header=SimpleNamespace(stamp=Stamp(12.2),
                                       frame_id="base_link")), Stamp(12.3))
        elif topic == "/tracking":
            yield (topic, SimpleNamespace(
                header=SimpleNamespace(stamp=Stamp(0),
                                       frame_id="odom")), Stamp(12.1))
            yield (topic, SimpleNamespace(
                header=SimpleNamespace(stamp=Stamp(12.3),
                                       frame_id="odom")), Stamp(12.4))

    def get_start_time(self):
        return 12.1

    def get_end_time(self):
        return 12.4


def test_rosbag_inventory_does_not_require_ros_or_assume_tf(tmp_path):
    bag = tmp_path / "recorded.bag"
    bag.touch()
    rows, overview = inventory_bag(
        bag, sample_per_topic=3, bag_factory=FakeBag)
    assert overview["read_only"]
    assert overview["topic_count"] == 2
    assert overview["topics_with_multiple_sampled_frames"] == 1
    points = next(r for r in rows if r["topic"] == "/lidar/points")
    assert points["ros_type"] == "sensor_msgs/PointCloud2"
    assert points["recorded_message_count"] == 3
    assert points["sampled_frame_ids"] == "base_link|lidar"
    assert points["status"] == "mixed_header_frames_needs_investigation"
    assert points["sampled_mean_record_minus_header_s"] == pytest.approx(.1)
    tracking = next(r for r in rows if r["topic"] == "/tracking")
    assert tracking["sampled_without_valid_header_stamp"] == 1
    assert tracking["status"] == "some_missing_header_stamps"
    output = tmp_path / "inventory.csv"
    write_inventory(rows, overview, output)
    assert len(list(csv.DictReader(output.open()))) == 2
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["topic_count"] == 2
    assert "no data were transformed" in metadata["warning"]


def test_rosbag_inventory_rejects_missing_files_and_bad_sample_count(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        inventory_bag(tmp_path / "x.bag", sample_per_topic=0,
                      bag_factory=FakeBag)
    with pytest.raises(FileNotFoundError):
        inventory_bag(tmp_path / "x.bag", bag_factory=FakeBag)
    with pytest.raises(ValueError, match="no topics"):
        write_inventory([], {}, tmp_path / "empty.csv")
