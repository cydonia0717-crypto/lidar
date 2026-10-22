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


## 第八阶段：两阶段速度/转角序列规划（新结构消融）

旧的单阶段候选规划只在0.4s执行后立刻推演刹停，难以表示让行和连续绕行等多步动作。新增 \`two_stage\`：生成第一段25组候选，并为每一组尝试少量第二段动作；对**延迟前缀→动作1→动作2→物理制动→停后6s**的整段道路和动态目标占据进行硬碰撞检查，选择最低代价的可行组合；只执行动作1，然后滚动重新规划。

\`\`\`bash
lidar-sim --scenario two_targets --mode two_stage --seed 0 --plot --decision-figure
lidar-eval --scenarios two_targets --stresses nominal,weak_brake \
  --modes braking_hold,two_stage --seeds 2 --max-time 30 \
  --output results/two-stage-ablation --print-episodes --plot
\`\`\`

详见 [第八阶段方法说明](docs/phase8_two_stage.md)。新算法依然是合成二维理想CV目标，**不是雷达实测也不是可证明安全算法**。仿真跑完必须检查成功、碰撞、超时和计算开销；两阶段效果不能预先保证。


## 第十阶段：双目标死锁的硬约束阶段归因

本阶段只增加逐阶段碰撞/越界拒绝日志，不修改\`two_stage\`规划选择。实际合成仿真 \`two_targets/nominal/seed0\` 已核实：20–30秒每周期平均只剩**5组可行两阶段停车类序列**、前进候选为0；第二阶段道路越界在20–25秒累计**5,200次候选拒绝**，同期动态目标/物理制动/停后占据拒绝均为0。注意累计拒绝事件不是车体碰撞次数，也没有证明真实物理车辆完全无路可走。

\`\`\`bash
python -m lidar_sim.rejection_diagnostics --scenario two_targets \
  --stress nominal --seed 0 --max-time 30 --output results/rejection-forensics
\`\`\`

真实运行结果与科学边界详见 [第十阶段归因实验结果](docs/phase10_findings.md)。下一步优先定位造成路界无解的\`y/yaw/steer\`轨迹、车体圆形近似与转向采样分辨率，而非继续添加奖励权重。


## 第十一阶段：道路边界的真实有向矩形与保守包围圆消融

第十阶段双目标 \`nominal/seed0\` 30秒超时，已定位为第二阶段道路越界拒绝而非障碍物拒绝。进一步按原1.9m车长、0.85m车宽复现实车体横向占据：静止位置 \`x=14.975m, y=1.964m, yaw=27.536°\`，原包围圆净空约0.045m、相同尺寸有向矩形净空约0.270m（均包含0.2m原安全余量）。

新增 \`two_stage_rectroad\`，仅把规划中的道路边界从固定整车包围圆替换成按实际航向旋转的车辆矩形；障碍物碰撞仍保守使用圆、路界余量和车辆尺寸不变。对照时新旧规划方法统一在**同一矩形道路物理真值**下测量出界与成功，保证公平。

\`\`\`bash
lidar-eval --scenarios two_targets --stresses nominal,weak_brake \
 --modes two_stage,two_stage_rectroad --physical-road-footprint rectangle \
 --seeds 2 --seed-start 0 --max-time 30 --output results/road-geometry-known --plot --print-episodes

lidar-eval --scenarios two_targets --stresses nominal,weak_brake \
 --modes two_stage,two_stage_rectroad --physical-road-footprint rectangle \
 --seeds 2 --seed-start 10 --max-time 30 --output results/road-geometry-holdout --plot --print-episodes
\`\`\`

详细模型与科学边界：[第十一阶段道路几何对照说明](docs/phase11_road_footprint.md)。所有性能收益需要实际跑完对照后确认；请勿将已验证的**几何净空改善**直接等同于已验证的**任务成功率改善**。


**已实跑阶段性结论**：[第十一阶段实际结果与其它场景外部验证](docs/phase11_pilot_results.md)：累计14组配对、28次合成闭环仿真，统一矩形物理路界真值：原包围圆规划成功8/14，新矩形规划成功13/14；两方法均碰撞1次，无实际道路越界。新方法明显减少了因边界保守约束造成的超时，但不能证明真实激光雷达/道路条件下的安全或统计显著性。


## 第十二阶段：后轴参考点与真实终点完成口径

上一阶段仅使用\`state.x >= goal_x - 0.6\`就记为成功，且有向矩形以自行车运动学状态点为中心。该状态的运动学实际上更适合按后轴中心解释，因此新建\`two_stage_bodyframe\`：在相同物理后轴参考下，后轴中心沿车头方向偏移假设的0.55m后得到车体中心，统一用于道路边界、目标距离和路线代价。0.55m是未经实车标定的工程假设。

独立的\`pose_gate\`终点口径要求车身中心越过x方向目标截面，横向偏差≤0.75m、航向偏差≤15°。其意义是“沿道路正确方向通过终点”，不是停车完成（无需零速度）。原\`x_gate\`和旧规划模式保持，不能拿此前旧判据的13/14与新的pose_gate成功次数直接混同。

\`\`\`bash
lidar-eval --scenarios two_targets --stresses nominal,weak_brake \
 --modes two_stage_rectroad,two_stage_bodyframe \
 --physical-road-footprint rearaxle_rectangle --goal-criterion pose_gate \
 --goal-lateral-tolerance 0.75 --goal-heading-tolerance-deg 15 \
 --seeds 2 --seed-start 0 --max-time 30 \
 --output results/bodyframe-known --print-episodes --plot
\`\`\`

详见 [第十二阶段：后轴车体中心和目标判据](docs/phase12_frame_goal.md)。本阶段仍是合成二维理想CV运动目标仿真。


**第十二阶段实际结果已完成**： [统一后轴参考点与严格终点位姿联合判据的实测报告](docs/phase12_pilot_results.md)。在相同rearaxle_rectangle物理真值、0.75m横向阈值、15°航向阈值下，8组配对旧状态点居中规划成功3/8、新后轴车体中心规划成功6/8；旧碰撞5次、新碰撞1次，但新方法弱制动仍出现1次碰撞和1次超时。这与第十一阶段13/14（原坐标约定+旧纵向过线规则）**不可直接横向比较**。


## 第十三阶段：后轴至车身中心偏置的标定误差敏感性

现有\`two_stage_bodyframe\`的0.55m车身偏置尚未经实车标定。为避免把**规划假设**与**仿真物理真值**混为一谈，新增彼此独立的\`assumed_body_offset_m\`和\`physical_body_offset_m\`：保持相同阿克曼底盘运动学、车长/宽、真实目标轨迹、控制延迟及完整制动约束，比较0.40/0.55/0.70m偏置误差的影响；保留全部成功/碰撞/超时和每个弱制动失败回合的逐帧几何净空、约束拒绝日志。

\`\`\`bash
python -m lidar_sim.offset_sensitivity --scenarios two_targets --stresses weak_brake \
 --seeds 0,1 --physical-offsets .40,.55,.70 --assumed-offsets .40,.55,.70 \
 --max-time 30 --output results/offset-grid

python -m lidar_sim.offset_sensitivity --scenarios two_targets --stresses weak_brake \
 --seeds 10,11 --physical-offsets .55 --assumed-offsets .40,.55,.70 \
 --forensic-seeds '' --max-time 30 --output results/offset-holdout
\`\`\`

所有结果来自二维合成CV目标仿真，无实车/3D雷达测量或保证。实验设计与边界详见 [第十三阶段研究方案](docs/phase13_offset_sensitivity.md)。


**第十三阶段已实跑**： [后轴车身偏置的3×3网格和弱制动失败机理报告](docs/phase13_results.md)。82项单测通过，24回合闭环合成仿真。物理偏置在0.40/0.55/0.70m、规划偏置在0.40/0.55/0.70m下，结果呈强非单调变化：\`assumed=0.55m\`在已知18回合中的对应6个回合全部失败，\`assumed=0.40m\`、\`0.70m\`各成功6/6；独立留出种子又得到0.55m两次碰撞。**这不能解释成应该把车辆“校准”为0.40m，而是暴露了离散轨迹优化的决策敏感性。**此外原seed0的30秒超时已在后期恢复移动、离目标只差0.15m；原seed1则在完全静止时被骑行目标侵入碰撞。


## 第十四阶段：超时时限与停车后动态目标侵入（仅诊断）

第十三阶段真实合成轨迹指出：\`two_targets/weak_brake/seed0\`在30秒时恢复前进且车身中心距目标线约0.15m，\`seed1\`在车速0时遭骑行目标侵入。新增无规划策略改动的\`horizon_forensics\`：比较30/35/40秒**实际重跑**结果、验证30秒前两条轨迹前缀数值一致；在停止位置计算合成CV目标静态12秒反事实物理重叠时刻与含margin/padding的规划包络入侵时刻，同时导出分阶段动作拒绝和5秒窗口。

\`\`\`bash
python -m lidar_sim.horizon_forensics --seeds 0,1 --horizons 30,35,40 \
  --assumed-offset .55 --physical-offset .55 --output results/phase14-forensics
\`\`\`

研究设计与科学边界见 [第十四阶段说明](docs/phase14_horizon_intrusion.md)。请不要把30秒末帧距终点近直接认定成功，也不能把固定位置风险视为递归安全保证。


**第十四阶段实跑结果**：[30/35/40秒截止时限与停驻侵入失效报告](docs/phase14_results.md)，独立CI 7项定向测试+6次实际闭环仿真通过。相同物理世界下种子0从30秒**超时**变为35/40秒时都在**30.10秒**真正满足pose_gate，且301帧30秒前前缀状态完全相同；种子1在三种时限下都于16.50秒**静止被骑行目标侵入**（物理保守圆净空−0.00456m）。种子1约9.7秒即无安全可行候选，约11.7秒起停稳并在反事实CV预测中可见后续固定车位侵入，15.2秒已提交控制前缀不安全。此阶段纯诊断，未改变规划动作或安全限制；结果不是实测LiDAR或实车保证。


## 第十五阶段：未来可行性探测（严格筛选之上的有限前瞻）

新增独立\`two_stage_viability\`模式：对原\`two_stage_bodyframe\`严格安全筛选通过的最多3个最佳**第一段**速度/转角，复制当前真实延迟队列并在0.60秒后执行同一基础规划器，查看是否仍有严格可行的动作序列。仅在原已可行候选中选择；如果所检查的候选未来均无解，明确标记并保持原方法选择，不绕过道路/障碍/全物理制动/停后6秒硬约束。它是有限深度诊断与动作选择启发式，**不是安全保证**。

实验设计与边界见 [第十五阶段方法](docs/phase15_viability.md)。已知weak_brake种子0/1和留出种子10/11采用35秒任务时限、相同物理后轴真值及严格姿态终点判据，额外比较独立crossing/narrow场景。


**第十五阶段已实跑消融**：[有限未来可行性探测与真实结果](docs/phase15_results.md)：7项针对性单元测试通过；6组配对共12次合成闭环仿真，双目标弱制动4个种子中原版1/4成功、3碰撞，新前瞻模式4/4成功、无碰撞；另外crossing/narrow两组均成功。动作首次分叉在2.5～2.9秒，每回合仅2～6次指令变化，却提前于旧模式的无解状态。但规划耗时在第二轮GitHub Runner约88～121ms/周期（首轮约130～162ms），原方法约18～22ms（首轮约26～34ms），**不能在100ms控制周期上声明实时性能**。这些种子属于小样本、部分已被以前阶段研究过，只有二维理想CV真值；没有概率安全或递归可行性保证。


## 第十六阶段：等价动作搜索提前退出

实验版\`two_stage_viability_fast\`保留第十五阶段严格可行候选及代价排序，仅将每个假设未来状态的**完整代价规划**替换为**找到任意完整严格可行续接即退出的布尔判据**，并在最优有续接第一段确定后停止其它候选探测。原\`two_stage_viability\`完整保留，逐帧对比**位置/航向/速度/实际控制指令**完全一致。不会削弱车体道路矩形、动态目标保守圆、完整制动、6s停驻占据、0.25s延迟及严格pose_gate要求。

验证命令：\`python -m pytest -q tests/test_fast_viability.py tests/test_viability.py\`；完整对照执行\`.github/workflows/phase16.yml\`。研究假设、CSV与局限见[第十六阶段计算优化方案](docs/phase16_fast_viability.md)。计算时间是GitHub Runner离线度量，尚非Jetson实测，未反馈到仿真控制延迟。


**第十六阶段已完成真实对照**：[逐帧完全等价的快速未来可行性搜索实验报告](docs/phase16_results.md)。独立专用CI已通过20项定向测试，16组配对（32次闭环合成仿真）中两方法全部安全完成、所有实际控制指令和车体姿态逐帧一致。GitHub Runner真实周期规划时间旧版总体均值111.64ms、P99 171.91ms，新fast版29.37ms、P99 50.64ms；100ms预算超限次数1795/2536→2/2536。**这不是Jetson实时测量，也没有将超时反馈物理控制延迟；没有严格实时/实车安全保证。**
