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


## 第一轮22秒闭环结果：开发回合与4个新增种子

原始数据：[成功工作流 Run #40081167262](https://github.com/cydonia0717-crypto/lidar/actions/runs/40081167262)，[seed0 Artifact #12555556099](https://github.com/cydonia0717-crypto/lidar/actions/runs/40081167262/artifacts/12555556099)，[seed50—53 Artifact #12555182559](https://github.com/cydonia0717-crypto/lidar/actions/runs/40081167262/artifacts/12555182559)。

| 场景（均为two_targets弱制动） | EMA | LS5 |
|---|---|---|
| 已知seed0：轻度退化 | 碰撞5.40秒 | 碰撞5.70秒 |
| 已知seed0：强退化 | 碰撞5.45秒 | 碰撞5.30秒 |
| 新seed50—53：轻度退化 | 0/4成功，4/4碰撞 | **2/4成功，2/4碰撞** |
| 新seed50—53：强退化 | 0/4成功，4/4碰撞 | 0/4成功，4/4碰撞 |

新seed50和seed52在`noisy_ls5`下分别于13.20s及13.10s完成任务；seed51、53依然碰撞。强退化下所有4组均碰撞，**即使测得速度误差有所改善，不能推断已具备安全裕度**。

仅对有有效轨迹的回合内决策时点计算平均速度误差，再等权对4个新seed求均值（非全时隙指标）：

| 条件 | EMA (m/s) | LS5 (m/s) |
|---|---:|---:|
| 轻度退化 | 0.8230 | 0.4467 |
| 强退化 | 1.8954 | 0.9974 |

上面分别约降低45.7%、47.4%，**但不同滤波器造成的不同闭环轨迹会改变目标观测距离，进而改变采样分布**。这组数字不是完全控制观测输入后的纯估计器效果。因此又加入下述固定自车轨迹实验，避免以混合了场景暴露差异的均值主张“纯滤波降噪45%”。

**证据强度**：4个新种子 + 一个已知困难开发种子，只能作为当前CV二维仿真分布的初步消融；不是普遍避障碰撞率、实车成功率或充分的统计显著性证据。不应将本结果包装为LS5总是安全改进。

## 固定自车轨迹的独立滤波精度对照（补充）

新增`phase18_velocity_offline.py`。对于seed50—53，所有估计器共同使用事先确定的**直线匀速1.5m/s自车轨迹**（并非真实闭环控制结果），传感器噪声与漏检仍使用相同case/tick/object的随机序列。程序逐帧严格断言两种估计器的有效目标ID、观测时间、带噪位置中心完全一致，才以独立真值评价：

- 速度估计的欧氏误差（m/s）；
- 使用该估计从当前规划时刻外推`0.5/1.0/2.0s`后的目标中心位置误差（m）；
- 由于无可靠观测而阻断的时隙及实际评分目标数。

输出`fixed_ego_pairs.csv`、`fixed_ego_residuals.csv`及可复核配置`inputs.json`。**此固定自车轨迹不受规划器影响、会穿过潜在碰撞目标，不能以其结果统计规划成功率或实际碰撞率**，它仅用于分离滤波算法本身的误差变化。

~~~bash
python -m lidar_sim.phase18_velocity_offline \
  --scenarios two_targets --stress weak_brake \
  --seed-start 50 --seeds 4 --ticks 161 --dt 0.1 \
  --output results/velocity-fixed-ego-50-53
~~~

为了避免对新种子过拟合，以上参数与试验字段已经固定；若根据结果改变窗口/速度平滑因子，必须在新的、此前未用于调参的种子上再做验证。未来实车也需要检验非CV机动目标和雷达掉点分布下的效果。
