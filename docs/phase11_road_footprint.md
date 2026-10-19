# 第十一阶段：有向矩形道路占据与圆形包络规划的配对消融

## 实测死锁车体状态：先查清再调整

来自 [GitHub Actions真实运行日志](https://github.com/cydonia0717-crypto/lidar/actions/runs/39219367016)，\`two_targets/nominal/seed0\`合成仿真：

| 5秒时段 | 结束x(m) | 结束y(m) | 结束航向角 | 结束转角(rad) | 最小包围圆道路净空(m) | 最小矩形道路净空(m) |
|---|---:|---:|---:|---:|---:|---:|
| 10–15s | 14.94 | 1.9454 | 28.29° | -0.42 | 0.0639 | 0.2801 |
| 15–20s | 14.975 | 1.964 | 27.536° | -0.42 | 0.0453 | 0.2699 |
| 20–25s | 14.975 | 1.964 | 27.536° | -0.42 | 0.0453 | 0.2699 |
| 25–30s | 14.975 | 1.964 | 27.536° | -0.42 | 0.0453 | 0.2699 |

道路半宽3.25m，原圆形车辆外包络半径 \`hypot(1.9/2,0.85/2)≈1.041m\`。两种几何使用相同车长1.9m、宽0.85m和安全边距0.2m。

原保守圆的宽度投影固定为半径 \(R\)。有向矩形沿道路横向的精确半宽为

\[
h_y(\theta)=\frac{L}{2}|\sin\theta|+\frac{W}{2}|\cos\theta|.
\]

道路的有符号安全余量 \(c_y=H-|y|-h_y(\theta)-m_{\mathrm{road}}\)。圆包络相应为 \(c_{disc}=H-|y|-R-m_{\mathrm{road}}\)。**矩形改变的是固定圆导致的过度保守近似，没有缩短真实车长、减小车宽或删除安全边距**。

但仅有22.5cm净空差并不能证明矩形规划就能成功：航向偏外约27.5°，车辆仍需向前转向才能摆正，动作可能仍超界。道路边界是这次死锁后期的主要约束来源，不代表所有场景都由同样因素主导。

## 控制试验

新增独立 \`two_stage_rectroad\` 模式：
- 基线 \`two_stage\` 的规划道路约束依旧使用保守包围圆。
- 新版 \`two_stage_rectroad\` 在规划的全部阶段使用**车辆实际朝向对应的有向矩形**作为道路边界占据检查。
- **其余全部相同**：25组第一段速度转角、第二段候选、阿克曼运动学与动力学、车辆尺寸、预测目标碰撞仍使用原保守圆、时空检查时刻、已经下发的延迟前缀、制动尾迹、停车后6秒占据检查，以及原评分权重。
- **物理真值必须公平**：对这两种方法的闭环结果都统一使用有向矩形车体作为真实道路越界检查；否则一个方法在圆形几何下认定越界、另一个在矩形下认定未越界，所得碰撞率/完成率不能直接对比。
- 障碍物碰撞仍采用保守圆；本阶段**只研究道路边界几何**，没有声称完成了全部车体与障碍物的精确矩形碰撞检测。

\`\`\`bash
pip install -e '.[dev]'
pytest -q

# 先导种子
lidar-eval --scenarios two_targets --stresses nominal,weak_brake \\
 --modes two_stage,two_stage_rectroad --physical-road-footprint rectangle \\
 --seeds 2 --seed-start 0 --max-time 30 \\
 --output results/road-geometry-known --print-episodes --plot

# 独立留出种子
lidar-eval --scenarios two_targets --stresses nominal,weak_brake \\
 --modes two_stage,two_stage_rectroad --physical-road-footprint rectangle \\
 --seeds 2 --seed-start 10 --max-time 30 \\
 --output results/road-geometry-holdout --print-episodes --plot
\`\`\`

可重复记录为\`episodes.csv\`，分别统计成功、碰撞、超时、越界和规划耗时。新增的\`paired_road_footprint_effects.csv\`逐回合差分并验证每对实验真的使用**同一物理矩形道路模型**，不允许偷换物理结果判据。

## 仍无法主张的事情

- 合成二维理想CV运动目标、无激光雷达误差/遮挡/跟踪误差；目标假设未覆盖行人突然变速/转向。
- 阿克曼状态\`(x,y)\`在旧项目中作为车体中心使用，矩形为**以相同参考点为中心**的近似；真实车轴至车体几何中心的偏置尚待标定。
- 采样间隔0.05s+包围圆障碍碰撞，不构成连续时间碰撞证明。
- 已知失败种子不等于独立测试集；即使有改进，亦需新增曲线道路、偏置目标和不同制动/延迟组合，以及蒙特卡洛复验。
- 两阶段规划在GitHub Runner上约几十毫秒，并非Jetson处理器实测；必须追踪计算时延变化。
