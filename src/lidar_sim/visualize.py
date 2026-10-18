"""Save figures from logged simulation states; never fabricate measurements."""
from __future__ import annotations

from pathlib import Path

from .experiment import Result
from .world import scenario


def save_figure(result: Result, path: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    world = scenario(result.scenario, result.seed)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), height_ratios=(2.5, 1), layout="constrained")
    ax, ax_v = axes
    ts = [item["time"] for item in result.trace]
    xs = [item["x"] for item in result.trace]
    ys = [item["y"] for item in result.trace]
    ax.axhspan(-world.road_half_width, world.road_half_width, color="#dfe8e1")
    ax.axhline(-world.road_half_width, color="black", ls="--", lw=1)
    ax.axhline(world.road_half_width, color="black", ls="--", lw=1)
    ax.axhline(0, color="gray", ls=":", lw=1, label="reference path")
    ax.plot(xs, ys, lw=2.3, label=f"ego: {result.mode}")
    ax.scatter(xs[0], ys[0], marker="s", label="start")
    ax.scatter(xs[-1], ys[-1], marker="x", label="last state")
    for obs in world.obstacles:
        ox, oy = zip(*(obs.position(t) for t in ts))
        ax.plot(ox, oy, ls="--", label=obs.name)
        ax.scatter([ox[0]], [oy[0]], s=20)
    ax.set(xlabel="x (m)", ylabel="y (m)", title=f"Synthetic {result.scenario} | {result.mode} | seed={result.seed}")
    ax.set_xlim(-1, world.goal_x + 3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8)
    ax_v.plot(ts, [i["speed"] for i in result.trace], label="ego speed")
    ax_v.set(xlabel="simulation time (s)", ylabel="speed (m/s)")
    ax_v.grid(alpha=0.2)
    ax_v.legend()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
