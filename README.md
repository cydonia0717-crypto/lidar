# 室外阿克曼小车动态避障：二维可复现仿真 (MVP)

研究用第一阶段原型：**不依赖 ROS / Gazebo / 实车**，用 Python 创建道路和动态障碍物，利用阿克曼运动学模拟小车，滚动采样目标速度与转角，并检查道路和时空碰撞约束，在可行候选中最小化轨迹代价。**当前输入采用仿真器给出的精确动态障碍物位置/速度，并不是激光雷达检测或真实 Kalman 跟踪数据**。

## 运行

需要 Python 3.10+：

~~~bash
python -m pip install -e '.[dev]'
lidar-sim --scenario crossing --mode predictive --seed 0 --plot
lidar-sim --scenario two_targets --mode reactive --seed 0 --plot
lidar-sim --scenario crossing --compare --seeds 10
python -m pytest -q
~~~

不安装 CLI 也可直接执行：

~~~bash
PYTHONPATH=src python -m lidar_sim.cli --scenario crossing --plot
~~~

默认将轨迹 CSV、指标 JSON 和（开启 --plot 时）PNG 写入 results/。--compare 对相同场景及相同随机种子分别运行两种规划模式，输出 paired_comparison.csv。不要把这些仿真结果当成实车数据。

## 已实现的内容

- 后轴中心参考的运动学自行车模型，约束最大速度、加速、制动减速度、前轮转角与转角速度。
- 道路边界、行人横穿、双目标交叉和狭窄道路；种子控制目标位置。
- 有限候选集 (target_speed, target_steer) 的 3 秒轨迹前推；动态目标在与车辆相同的未来时间戳上按 CV 模型预测。
- 用保守车体包围圆与目标圆检测碰撞；采样时间步内尚未进行解析连续碰撞检验，不能提供形式化安全保证。
- 目标函数包含参考路线偏移、纵向前进量、速度改变以及转向改变惩罚；无可行候选时请求制动，**不保证任何情况下都来得及停下**。
- reactive 模式将障碍物固定在本周期当前的位置；predictive 模式使用已知 CV 运动模型按未来时间进行碰撞检测（并增加一个示例额外余量）。这仅是最简单的对照，不代表全面文献基线。
- 轨迹 CSV、指标 JSON、轨迹/速度曲线 PNG、pytest 和 GitHub Actions。

## 配置和指标说明

参数在 src/lidar_sim/model.py 的 VehicleConfig、planner.py 的 PlannerConfig 与 experiment.py 的 RunConfig 中。当前数值为演示性质的**假设车辆参数**，未来需要进行底盘标定和参数敏感性分析。起点 (0,0)，前进方向为 +x，道路中心线 y=0。小车几何暂时用保守包围圆替代矩形旋转碰撞体。

- success：到达目标纵向坐标（尚未加入严格的末端横向/航向条件）。
- collision：仿真实际时刻与障碍物包围圆发生重叠。
- road_exit：车体保守包围圆超出道路。
- min_clearance_m：过程中车体与障碍物/边界的最小净距离，负值代表穿透。
- emergency_steps：所有候选都不满足安全检测而退化到紧急制动指令的周期数。
- duration_s：到达、碰撞、越界或超时前经过的仿真时间；仅成功的回合应参与平均通行时间比较。
- mean_planning_ms：当前计算机上各周期 choose_action 的平均墙钟时间，不代表 Jetson 实时性能。

## 近期开发任务（尚未实现）

1. 实验基线完善：在相同安全余量和运动约束下统一比较，增加更复杂的突然转向/停止场景、完整消融及置信区间。
2. 数据链：仿真带噪测量、漏检、Hungarian 关联、Kalman-CV 跟踪，严格区分真值、量测、估计与预测。
3. 不确定性：独立轨迹校准/测试划分，覆盖率、误差包络面积和按观测质量分组的可靠性检验；不预设一定优于固定半径。
4. **重点研究：控制延迟前缀 + 候选动作执行 + 后备制动尾迹**，检查全部时段的车辆扫掠区域与动态目标占据区域/道路边界；实测前暂用假设制动参数做敏感性分析。
5. 选做：车辆矩形碰撞几何、连续时间碰撞近似、自适应采样、其他标准规划基线、ROS/Gazebo接入与后续实车验证。

所有实验结论仅限于明确的仿真模型与场景。仿真通过不等于真实环境中的安全性或论文创新已经得到验证。

## 第二阶段：控制延迟与完整制动尾迹（第二个开发分支）

已增加 `src/lidar_sim/safety.py`：将延迟队列中还未生效的指令、候选速度/前轮转角、以及按物理制动减速度刹停的后备轨迹串联，用同一时间轴检查移动目标与道路边界。仿真执行也使用同一控制队列，避免规划过程与执行模型不一致。

```bash
lidar-sim --scenario crossing --compare --seeds 10
lidar-sim --scenario two_targets --mode braking --plot
lidar-sim --scenario late_crossing --mode braking --plot --backup-figure
lidar-sim --scenario crossing --compare --seeds 10 --control-delay .5 --deceleration 1.0
```

模式 `predictive`：未来障碍物预测但忽略控制队列；`delay_aware`：加入控制延迟前缀；`braking`：进一步加入候选执行结束后的完整制动尾迹；`reactive`：冻结目标当前位置。

日志额外输出 `backup_rejections`、`unsafe_fallback_steps`、`nominal_no_backup_steps`：全部候选没有安全后备方案时**不得声称安全保证**，因为在某些移动目标场景中停车本身也会发生碰撞。当前仿真使用理想目标运动真值，尚无雷达观测误差与实车标定。详见 [第二阶段算法与实验说明](docs/phase2_braking_filter.md)。


## 第三阶段：可解释的自身运动决策

不再只输出“选中了某条轨迹”：每个控制周期把25组候选目标速度/前轮转角展开，先做道路、动态障碍物和（braking模式）完整后备制动硬约束检查，再比较**路线偏差、沿线前进量、速度变化、转向变化、航向偏差、安全净空**六项加权成本。

所有正常动作的约束状态、总成本和选中标志都能在CSV中审计；闭环 trace 记录被选择动作的各成本分项。可以直接生成组会需要的决策对比图（计算得分，不是效果假想图）：

```bash
lidar-sim --scenario crossing --mode braking --seed 0 --plot --decision-figure
lidar-sim --scenario crossing --mode braking --seed 0 --clearance-weight 2.0 --decision-figure
pytest -q
```

新增的文件是 results/ 下 `*_decision_candidates.csv` 和 `*_decision.png`。不同模式的安全 padding 统一，防止对照时因安全余量不同而误判提升。

**仍未完成的论文部分**：带噪声的目标跟踪/漏检、一般曲线参考路线、目标预测误差校准、闭环独立测试的制动安全收益验证。具体数学公式与科学边界详见 [决策层算法说明](docs/phase3_decision_layer.md)。


## 第四阶段：仿真配对对照与参数敏感性实验

增加独立命令 `lidar-eval`，按**同一目标轨迹、同一车辆参数、同一延迟、同一场景随机种子**运行多个决策方法，并改变单一压力因素：控制延迟、刹车能力、动态目标速度、道路宽度。

```bash
# 快速验证：2场景×2压力×3种子×3方法=36次仿真
lidar-eval --scenarios crossing,two_targets \
  --stresses nominal,weak_brake --modes predictive,delay_aware,braking \
  --seeds 3 --output results/eval_quick --plot

# 常规评估：默认4场景×5压力×5种子×3方法=300次仿真
lidar-eval --output results/eval_main --plot
```

输出原始回合 `episodes.csv`、分层 `summary_by_scenario_stress_mode.csv`、同场景差分 `paired_braking_effects.csv`、全部实验假设 `metadata.json`、成功/碰撞率对比图 `rates_by_stress.png`。Wilson 95%区间仅是简单描述统计，不能忽视场景之间的相关性或仿真模型误差。尤其不能只凭3~5组种子做论文中的显著结论。

详见 [第四阶段实验设计](docs/phase4_paired_sensitivity.md)。本项目当前仍然**只包含理想CV目标的人工合成数据**，不存在真实雷达实测和安全性形式化保证。


### 第四阶段先导实验与失败回合追踪

GitHub Actions 已实际运行 **36次配对合成仿真**。在单人横穿场景，三种模式全部通过；在双目标场景，加入制动筛选的 `braking` 并未比 `delay_aware` 改善成功/碰撞结果。请阅读 [实际先导实验结果与失败诊断计划](docs/phase4_pilot_findings.md)，不要把尚未验证的创新收益写成已经实现。

批量程序现在默认保存每个场景×压力×方法最多2个失败回合的完整 CSV 状态轨迹到 `failure_traces/`，使用 `--failure-traces-per-group 0` 可以关闭。重点是分析失败发生在哪里、当时还有哪些可用动作、何时进入了没有经过安全验证的后备退化状态。


## 第五阶段：动态目标撞向停车位置的事故复现与后停驻占据检查

自动化诊断已复现并报告：双目标回合中，车辆约10—12秒进入无可行候选动作的紧急制动退化状态，随后保持静止，被经过的骑行者碰撞。原 \`braking\` 只检查至车辆刹停、无法识别几秒后到来的目标。

新增消融 \`braking_hold\`：在原三段运动后，继续检验静止停车位置在未来6秒内是否与动态目标的CV轨迹冲突；原 \`braking\` 原样保留。参数 \`--stationary-hold\` 可控制检查时间，传0秒时必须与原制动筛选一致。

\`\`\`bash
lidar-sim --scenario two_targets --mode braking_hold --stationary-hold 6 --plot --decision-figure
lidar-eval --scenarios two_targets --stresses nominal,weak_brake \
 --modes braking,braking_hold --seeds 3 --max-time 30 \
 --output results/post-stop-ablation --plot --print-episodes
\`\`\`

实际先导对照详情见 [第五阶段方法及边界](docs/phase5_post_stop_viability.md) 与 [第五阶段初步实验报告](docs/phase5_pilot_results.md)。目前仅属于合成仿真改进。新增静止检查并没有获得全面更好的成功率，可能增加等待时间和保守性。
