# 第九阶段：两阶段规划的时间不一致性和停车超时诊断

## 为什么继续

第八阶段真实合成数值仿真（GitHub Actions Run 39210251048）发现：

- nominal种子0：单阶段\`braking_hold\`在19.25秒碰撞，两阶段\`two_stage\`不碰撞但30秒超时；
- nominal种子1：两阶段成功，14.0秒，单阶段17.4秒；
- weak_brake种子0：两种方法均30秒超时；
- weak_brake种子1：两阶段成功13.4秒，单阶段17.8秒。

先导4对试验中，完成率没有提高。平均规划耗时，单阶段约9–10ms/周期，两阶段约35–36ms/周期（GitHub Actions主机；不是Jetson上的实测）。

## 待验证的失效机制：第二段进度“虚假承诺”

\`two_stage\`在候选路径内计划\`first\`运行0.4秒、随后\`second\`运行1.0秒，然后物理刹停并检查停后6秒动态目标占据。但闭环每0.1秒重新调用规划器，而且每周期**只把first发给延迟队列**。

因此可能产生时间不一致：每次都选择“第一段等待、第二段推进”的序列，重新规划后第二段不断推迟。需要通过实际trace确认这种行为是否在超时回合出现，不能只从算法结构推断原因。

### 新增诊断（实际闭环状态）

\`python -m lidar_sim.stall_diagnostics\`把实际执行记录按5秒窗口统计：

- 真实自车x方向位移和平均速度；
- 第一阶段目标速度为0的指令比例；
- 车辆静止时间比例、紧急状态比例；
- 最优规划第二阶段的目标速度及**预期前段/后段进度**（两者是预测的，不是真执行）；
- 原始每个控制周期中的执行指令仍保留于trace。

用这些指标可区分：
1. 反复规划“先停后动”（第一段长期速度0，第二段计划正向且车辆x几乎不变）；
2. 硬约束已无解（\`emergency_brake\`比例高）；
3. 低速沿道路缓慢前进或沿弯线不再正向增加x等其他原因。

## 独立消融：two_stage_commit

**不修改原有\`two_stage\`，也不改变其名义动态碰撞、道路边界、物理刹车和6秒停后占据硬约束。** 候选动作序列、执行时长和动力学完全一致。新增的唯一目标函数项：

\[
J_\mathrm{first}(u_1,u_2)=-w_f\,
\max\big(0,\min(x_\mathrm{first},x_g)-\min(x_\mathrm{prefix},x_g)\big).
\]

其中\`x_first\`是延迟前缀结束后执行**当前可提交的第一段动作0.4秒**形成的预测x位置；\`x_prefix\`是已提交延迟前缀终点；\`x_g\`是沿道路x轴方向的终点；\`w_f=7.0\`是预先设定的实验参数，不是从已知碰撞种子中优化得到的最佳权重。

\`J_total = J_original + J_switch + J_first\`，仅**严格通过所有硬约束的候选序列**才可比较；在全无解时仍明确触发受物理模型约束的退化停车请求。该项是对*第一段执行中的前进量*的软偏好，不能防止所有长期停车或目标突然转向。

关闭\`w_f=0\`时，\`two_stage_commit\`与原\`two_stage\`必须返回完全相同的最优动作及可行候选数量。每一步trace新增\`planned_first_progress_m\`、\`planned_second_progress_m\`、\`planned_second_speed\`、\`cost_first_stage_commitment\`，从而能够判断它到底把未来的动作提前了多少。

## 预先规定的对照方案

\`\`\`bash
pip install -e '.[dev]'
pytest -q

# 第八阶段使用过的回归种子；只用于定位
lidar-eval --scenarios two_targets --stresses nominal,weak_brake \\
 --modes two_stage,two_stage_commit --seeds 2 --seed-start 0 \\
 --max-time 30 --output results/commit-known --print-episodes --plot

# 明确留出的独立种子10、11，尚未参与方法设计
lidar-eval --scenarios two_targets --stresses nominal,weak_brake \\
 --modes two_stage,two_stage_commit --seeds 2 --seed-start 10 \\
 --max-time 30 --output results/commit-holdout --print-episodes --plot

# 以5秒为窗口检查两阶段动作的真实执行/未来承诺
python -m lidar_sim.stall_diagnostics --scenario two_targets \\
 --stress nominal --seed 0 --mode two_stage \\
 --output results/stall-original
\`\`\`

须逐种子同时报告成功、碰撞、超时、出界、成功回合行驶时间和每周期计算耗时；不能将“延迟碰撞转为超时”声称是成功率改善。

## 科学限制

1. 模型仍使用合成二维恒速度目标**理想真值**，尚非16线雷达带噪观测、漏检、Hungarian数据关联和CV-KF。
2. 车辆包围圆仅是保守近似，离散采样不能得出连续时域无碰撞证明。
3. 第二段始终是预测，并未保证在下一次控制周期能够落实。新软项不构成递归可行性保证。
4. 先导独立种子只有2个/压力，必须扩大测试并增加目标加速/转向、道路曲率、不同延迟等分层；不能用这些数据宣称显著效果。
