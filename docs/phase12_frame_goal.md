# 第十二阶段：阿克曼运动学参考点与“真正到达终点”判据修正

## 为什么必须做

此前第十一阶段的合成仿真使用车体有向矩形道路边界后，14组配对实验中新方法13/14成功、旧方法8/14成功。但现有成功定义只有\`state.x >= world.goal_x - 0.6\`，没有检查横向偏差与车头方向。且自行车运动学状态\`(x,y)\`使用后轴中心的运动方程，而车体矩形直接以这个\`(x,y)\`为几何中心。**之前的13/14属于原判据，不得自动引用为新判据的成功率。**

本阶段首先纠正评价口径，并将后轴到车身中心的偏置显式建模。仿真继续使用理想合成CV目标；这里是模型一致性工程工作，不宣称算法理论创新。

## 模型

### 车体中心

令后轴中心位姿\((x_r,y_r,\psi)\)，车身几何中心距离后轴沿车辆前向\(\ell_c\)，则

\[
x_c=x_r+\ell_c\cos\psi,\qquad y_c=y_r+\ell_c\sin\psi.
\]

当前\(\ell_c=0.55m\)是**未经底盘测量的假设值**，真实阿克曼底盘需重新标定，避免把模型参数误称真实测量结果。车辆尺寸仍为长1.9m、宽0.85m，轴距1.35m，矩形横向半宽仍按\(\frac{L}{2}|\sin\psi|+\frac{W}{2}|\cos\psi|\)计算。动态障碍物保守圆的中心、道路外廓矩形中心、路径路线偏差和目标前进量，在新模式下都使用变换后的车身几何中心；车辆运动学仍积分后轴中心状态。

### 新终点判据

历史\`x_gate\`：只看状态参考点是否满足\`x >= goal_x - 0.6m\`。

新增\`pose_gate\`：同时满足

- **车身中心**纵向过目标横截面：\`body_center_x >= goal_x - 0.6m\`；
- **横向偏差**：\`|body_center_y| <= 0.75m\`；
- **航向误差**：\`|wrap(yaw)| <= 15°\`；
- 这是道路**穿越终点**的定义，因此不默认要求速度=0；如果目标改为停车入位，可额外设置速度上限，届时必须重做实验。

任何物理道路越界或障碍物实际碰撞都不计为成功。trace和指标分别保留\`reached_historical_x_line\`与\`reached_pose_gate\`，防止将原来的纵向过线率与新终点完成率混为一谈。

## 独立消融和公平真值

- 基线\`two_stage_rectroad\`：保留第十一阶段“状态点即矩形中心”的规划假设；
- 新\`two_stage_bodyframe\`：在**相同两阶段动作、控制延迟、速度转向限制、物理制动、停后6秒约束和参数权重**下，规划中的车体矩形道路、动态目标保守圆和路线成本都参考真实后轴偏置后的车身中心；
- 两方法闭环都使用相同的物理参考模型\`--physical-road-footprint rearaxle_rectangle\`来检查道路/动态目标碰撞，且都使用同一\`--goal-criterion pose_gate\`判定成功，避免标准不一致；
- 独立输出\`paired_bodyframe_effects.csv\`并校验物理真值、终点阈值等相同；新增\`body_center_x/y\`、\`abs_heading_error_deg\`等逐帧字段。
- 本阶段改变的因素是**参考坐标一致性**；几何、路径成本相应同步变更，因此不应把结果当成“只改一个代价项”的消融。

## 实验命令

\`\`\`bash
pip install -e '.[dev]'
pytest -q
lidar-eval --scenarios two_targets --stresses nominal,weak_brake \\
 --modes two_stage_rectroad,two_stage_bodyframe \\
 --physical-road-footprint rearaxle_rectangle --goal-criterion pose_gate \\
 --goal-lateral-tolerance 0.75 --goal-heading-tolerance-deg 15 \\
 --seeds 2 --seed-start 0 --max-time 30 \\
 --output results/bodyframe-known --print-episodes --plot

lidar-eval --scenarios crossing,narrow --stresses nominal \\
 --modes two_stage_rectroad,two_stage_bodyframe \\
 --physical-road-footprint rearaxle_rectangle --goal-criterion pose_gate \\
 --seeds 2 --seed-start 30 --max-time 30 \\
 --output results/bodyframe-other-scenarios --print-episodes --plot
\`\`\`

不事先假定新模型成功率更高。因为旧规划对车身中心有不同假设，在统一物理真值下，结果可能更好、更差或出现新的碰撞。无论如何必须如实报告。已有第十一阶段结果只作为**旧定义下的背景实验**，不可和新阶段不同的物理真值直接算同一配对。

## 尚未完成的研究

- 0.55m偏置需要真实车辆尺量，动力学轮胎侧偏尚未建模；
- 合成目标使用真实CV状态，没有LiDAR点云噪声、遮挡、Kalman/Hungarian跟踪；
- 道路是笔直x轴方向的理想走廊，新终点判据也基于这个特定参考路径，未来应支持参考曲线的切向/法向投影及末端姿态；
- 动态障碍仍用保守包围圆、离散碰撞采样，缺少连续时间安全保证；
- 通行到终点时并不要求速度为零，不能将pose_gate结果描述为停车入位性能；
- 小规模模板化仿真不是独立统计大样本结论。
