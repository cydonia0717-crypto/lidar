"""Plot recorded simulated trajectories; no invented data or paper-result reuse."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    a = p.parse_args()
    local_runtime = Path(__file__).resolve().parents[1]/'plot_runtime'
    if local_runtime.exists():
        sys.path.insert(0, str(local_runtime))
    os.environ['MPLCONFIGDIR'] = str(a.results/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    for path in [Path('C:/Windows/Fonts/msyh.ttc'), Path('C:/Windows/Fonts/simhei.ttf')]:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10, 'svg.fonttype': 'path'})
    data = json.loads((a.results/'social_results.json').read_text(encoding='utf-8'))
    rows = list(csv.DictReader((a.results/'social_observations.csv').open(encoding='utf-8')))
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), constrained_layout=False)
    names = {'success': '成功', 'collision': '碰撞', 'timeout': '超时', 'execution_error': '执行异常'}
    for ax, episode in zip(axes, data['episodes']):
        n = episode['initial_targets']
        group = [r for r in rows if int(r['initial_targets']) == n]
        for identity in sorted({int(r['id']) for r in group}):
            target = [r for r in group if int(r['id']) == identity]
            ax.plot([float(r['x']) for r in target], [float(r['y']) for r in target], color='#aab4c0', alpha=.65, lw=.65)
            ax.scatter(float(target[0]['x']), float(target[0]['y']), color='#8896a6', s=13, zorder=2)
        ego = {}
        for r in group:
            ego[float(r['time_s'])] = (float(r['robot_x']), float(r['robot_y']))
        points = [ego[t] for t in sorted(ego)]
        if points:
            ax.plot([0.]+[v[0] for v in points], [0.]+[v[1] for v in points], color='#c53a3a', lw=1.8, zorder=4)
            end_color = {'success': '#278351', 'collision': '#a72121', 'timeout': '#c88420'}.get(episode['outcome'], '#666666')
            ax.scatter(*points[-1], color=end_color, edgecolors='white', s=42, zorder=5)
        ax.plot([-6.5, 39.5], [3.5, 3.5], color='#25303b', lw=1.1)
        ax.plot([-6.5, 39.5], [-3.5, -3.5], color='#25303b', lw=1.1)
        ax.axvline(40, color='#278351', ls='--', lw=.8)
        ax.scatter(0, 0, color='#c53a3a', s=22, zorder=5)
        ax.set(xlim=(-7, 43), ylim=(-4.3, 4.3), ylabel='横向位置 / m')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('%d人走廊｜%s｜仿真时间 %.2f s｜20 Hz采样最小行人净空 %.3f m' %
                     (n, names.get(episode['outcome'], episode['outcome']), episode['simulation_time_s'], episode['min_sampled_ped_gap_m']), loc='left', fontsize=10)
        ax.grid(alpha=.16, linewidth=.5)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel('纵向位置 / m')
    fig.suptitle('DRA-MPPI复现：作者社会力人群与兼容桥接的记录轨迹', fontsize=14, y=.98)
    fig.legend(handles=[Line2D([0], [0], color='#c53a3a', lw=1.8, label='小车记录轨迹'),
                        Line2D([0], [0], color='#aab4c0', lw=1, label='行人记录轨迹'),
                        Line2D([0], [0], color='#278351', ls='--', lw=1, label='40 m完成线')],
               loc='upper center', bbox_to_anchor=(.5, .945), ncol=3, frameon=False)
    fig.text(.5, .025, '合成仿真；固定种子5；K=64，T=20，MC=2048；自建独轮车桥接，非Gazebo、实车或论文统计表复现。',
             ha='center', fontsize=9, color='#505863')
    fig.subplots_adjust(top=.87, bottom=.09, left=.08, right=.98, hspace=.43)
    for suffix in ['png', 'svg']:
        fig.savefig(a.results/('社会力走廊记录轨迹.'+suffix), dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
