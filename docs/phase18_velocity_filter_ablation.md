# 第18阶段：因果速度估计对闭环避障的消融研究

## 研究动机与定位

仿真是当前主要研究与验证方式，今后可能转回Jetson/ROS1/真实底盘，所以估计状态数据结构仍保留时间戳、局部`odom`坐标系及目标位置/速度/保守半径。**当前无需真实数据或ROS bag。**

此前[因果感知闭环](phase18_simulation_perception.md)的`two_targets/weak_brake/seed0`出现：`noisy`配置下两种规划器同在5.40秒物理碰撞；已有离线真值评价显示有检测时速度估计误差可能达到约0.81m/s，而位置信息噪声比较小。要回答的新问题是：**相同带噪中心观测条件下，单独更换速度滤波算法是否改变闭环结果？**

## 公平对照和可证伪性

`PerceptionConfig.velocity_estimator`增加严格因果的两种实现：

- `ema`：原第18阶段相邻已到达观测中心的差分速度，首两帧直接差分，之后按`velocity_update_gain=0.5`对速度指数平滑；历史算法**保留默认不变**。
- `windowed_ls`（命名`ls5`）：最多5个**已经观测到**的中心点，以**最早观测的时间戳为零点**做时间—位置一元最小二乘回归估计`vx,vy`。可设置`regression_window`，不使用目标真值速度、未来观测或后验参考标签。5点是工程候选配置，没有声称已理论最优。

两种滤波器都用**最新一次中心观测**作为预测位置锚点；区别只在速度估计，所以不能把任何中心位置精度改善归因于LS5直接改变了位置测量。LS5在恒速、独立位置高斯噪声下有潜在降噪收益，但目标突然加速、转弯或长时间掉检时，长回归窗口也可能造成响应滞后，增加碰撞或超时；因此需如实报告**负收益案例**。

| 原始观测环境 | EMA baseline | 仅改变速度估计 |
|---|---|---|
| 轻度误差+5%丢检+1 tick数据延迟 | `noisy` | `noisy_ls5` |
| 强噪声+25%丢检+2 tick数据延迟 | `degraded` | `degraded_ls5` |

两者的位噪声幅度、漏检概率、量程、视野、捕获时延、随机数种子及目标真值完全相同。随机测量是按`case_id/tick/target_id`稳定生成的。须注意**闭环差异可能导致后续车辆位置和视野不一样**，进而改变哪些观测被实际提供；共享随机机制不是对每时刻观测数据逐帧完全相同的保证。若需要纯估计器精度比较，应在相同的固定真值ego历史上离线执行两个滤波器，以消除闭环感知分布的内生变化。

### 验收设计

- **因果性单元测试**：首次观测不能使用真值速度；无噪恒速CV在两次不同时间的观测后可重建；相同测量与seed使EMA/LS5中心位置完全相同；2点LS等价于原始差分；历史窗口不能包含未来；大ROS Unix时刻数值可用；所有非法窗口/模式 fail-fast。
- **已知开发诊断**：`two_targets/weak_brake/seed0`、固定22秒上限、只比较第16阶段`two_stage_viability`规划器、4个profile（2对）。
- **提前冻结的新种子**：`two_targets/weak_brake/seed50,51,52,53`，同一22秒上限，四profile、共16回合、8对。**这些是在同一类困难场景下的新增仿真种子，不是外部独立分布或真实测试集。** 不得把它们的碰撞比例泛化为机器人安全可靠性。
- 输出`episodes.csv`、每回合全轨迹与`phase18-velocity-paired.csv`配对报告。结果必须包括成功、碰撞、道路越界、超时、完成时长、最小物理净空、感知不可用步数，以及估计误差可评分的决策时隙数量。
- 本次比较**不增加预测包络**也**不更改未来可行性筛选**，以便分离单个速度估计因素。后续是否研究位置/速度不确定性安全包络，要根据这些对照是否有实质作用再决定。

运行方式：

~~~bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_phase18_velocity_filters.py tests/test_phase18_velocity_analysis.py

python -m lidar_sim.phase18_sim_eval \
  --scenarios two_targets --stress weak_brake --seed-start 50 --seeds 4 \
  --modes two_stage_viability \
  --profiles noisy,noisy_ls5,degraded,degraded_ls5 \
  --max-time 22 --output results/velocity-validation-50-53

python -m lidar_sim.phase18_velocity_analysis \
  --root results/velocity-validation-50-53 \
  --output results/velocity-validation-pairs.csv
~~~

### 对数据指标的严格解读

有效目标误差基于仿真内部真实位置/速度，只能**在规划决策之后**计算。其`*_scored_ticks`小于`decision_ticks`时，未评分时隙不能补0，也不能将只在目标存在时的均值当作总感知质量。碰撞和成功必须使用真值物理状态，严格区分停车超时与目标抵达。

当前传感器是二维中心点噪声模型（LiDAR-like），**不具有真实3D PointCloud2、射线遮挡、点数密度、错误匹配/ID切换、假阳性、地面反射或精确底盘姿态误差**；目标ID/半径及自车位姿仍采用真值，转向与执行器参数尚未实车标定。滤波器表现只支持本合成CV场景的结论。

若以后回到实车，EMA/LS5应作用于同一`odom`参考系、同一时钟标准下的连续目标检测/跟踪估计，结合ROS输出中真正的跟踪可见性和独立安全看门狗；**当前不向底盘发布命令，不宣称跨平台已完成移植**。
