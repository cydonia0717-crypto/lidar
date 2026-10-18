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


def save_backup_figure(initial, command, line, world, vehicle, planner, path):
    """Illustrate one computed prefix/action/brake rollout, not a fabricated photo."""
    from .safety import backup_segments
    from .planner import collision_free
    from matplotlib.patches import Circle
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    segs = backup_segments(initial, command, line, vehicle,
                           planner.candidate_execute_s, planner.safety_dt)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.axhspan(-world.road_half_width, world.road_half_width, color="#e4ebe9")
    ax.axhline(-world.road_half_width, color="gray", ls="--")
    ax.axhline(world.road_half_width, color="gray", ls="--")
    ax.axhline(0, color="gray", ls=":", alpha=.5)
    for points, color, name in ((segs.prefix, "#3a6ca8", "committed delay prefix"),
                                (segs.candidate, "#159878", "candidate execution"),
                                (segs.brake, "#e18f35", "physical braking tail")):
        ax.plot([s.x for s in points], [s.y for s in points], color=color,
                lw=4, marker=".", ms=3, label=name)
        ax.scatter([points[-1].x], [points[-1].y], color=color, s=48, zorder=5)
    for obs in world.obstacles:
        ox = [obs.position(s.time)[0] for s in segs.complete]
        oy = [obs.position(s.time)[1] for s in segs.complete]
        ax.plot(ox, oy, color="#cf5956", ls="--", lw=2, label=f"{obs.name} actual/CV")
        stride = max(1, len(ox) // 10)
        for j in range(0, len(ox), stride):
            ax.add_patch(Circle((ox[j], oy[j]), radius=obs.radius,
                                fc="#cf5956", ec="none", alpha=.10))
    valid = collision_free(segs.complete, world, vehicle, padding=planner.prediction_padding)
    ax.scatter([initial.x], [initial.y], marker="s", s=80, color="#1b365e", label="ego start")
    ax.set(xlabel="x (m)", ylabel="y (m)",
           title=f"Computed 3-stage candidate | feasible={valid} | delay={line.delay_s:.2f}s, "
                 f"execute={planner.candidate_execute_s:.2f}s")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8, loc="best")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
