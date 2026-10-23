"""Phase17 replay must reproduce plant dynamics independent of CPU speed."""
from dataclasses import replace

import pytest

from lidar_sim.experiment import RunConfig, run
from lidar_sim.phase17_replay import compare_rows, recorded_delays


def as_text(trace):
    return [{k: ("" if v is None else str(v)) for k, v in row.items()}
            for row in trace]


@pytest.mark.parametrize("delay_ms", [0, 80, 160, 250])
def test_fixed_delay_trace_replay_is_physically_identical(delay_ms):
    cfg = RunConfig(max_time=.7, compute_delay_mode="fixed",
                    compute_fixed_ms=delay_ms)
    original = run("empty", mode="two_stage_bodyframe", config=cfg)
    captured = as_text(original.trace)
    times = recorded_delays(captured)
    replay = run("empty", mode="two_stage_bodyframe",
                 config=replace(cfg, compute_delay_mode="replay",
                                compute_replay_ms=times))
    assert compare_rows(captured, replay.trace) < 1e-8
    assert original.missed_control_ticks == replay.missed_control_ticks
    assert original.submitted_commands == replay.submitted_commands


@pytest.mark.parametrize("delays", [(), (-1.0,), (float("nan"),),
                                      (float("inf"),)])
def test_bad_replay_trace_rejected(delays):
    with pytest.raises(ValueError, match="replay"):
        run("empty", config=RunConfig(
            max_time=.3, compute_delay_mode="replay",
            compute_replay_ms=delays))


def test_replay_fails_closed_if_source_has_too_few_decisions():
    with pytest.raises(ValueError, match="exhausted"):
        run("empty", mode="two_stage_bodyframe",
            config=RunConfig(max_time=.4, compute_delay_mode="replay",
                             compute_replay_ms=(0.0,)))


def test_compare_rows_rejects_changed_issued_command():
    a = {"time": "0.0", "command_speed": "2.4"}
    b = {"time": 0.0, "command_speed": 1.2}
    with pytest.raises(AssertionError, match="command_speed"):
        compare_rows([a], [b])


def test_recorded_delays_skips_busy_and_terminal_ticks():
    rows = [{"compute_busy": "0", "compute_submit_delay_ms": "160.0"},
            {"compute_busy": "1", "compute_submit_delay_ms": ""},
            {"compute_busy": "", "compute_submit_delay_ms": ""}]
    assert recorded_delays(rows) == (160.0,)
