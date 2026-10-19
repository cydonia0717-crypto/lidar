# 第七阶段：停车终端安全净空偏好（不放松任何硬约束）

## 为什么要做

第六阶段的\`recovery\`在双目标测试中反而更差：两种压力下的6个配对场景，总成功回合从2降为0，碰撞从3增至5。这说明**严格可行集已经耗尽以后再用预测风险排名冒险执行，并不是可靠死锁解法**。保留这次失败实验作为消融，不将其纳入正式主线。

因此转向更早阶段的规划决策：当存在多个**已经满足完整制动+停后占据安全约束**的候选时，选更不容易停在接近动态目标运动路线或道路边界的位置，以降低进入未来不可行动区域的倾向。

## 核心算法

\`terminal_margin\`与\`braking_hold\`使用完全一致的：

- 阿克曼自行车动力学、物理速度/转向/转向速率限制
- 控制延迟前缀和之前已排队的控制指令
- 3秒名义轨迹的动态碰撞与道路边界硬检查
- 候选动作0.4秒执行+全物理制动尾迹+停后6秒占据的硬检查
- 碰撞圆、道路边界和全部假设参数

当候选动作未通过任何一个硬检查时，**直接拒绝**。只有通过全部硬条件的动作才计算新的终端净空偏好：

设\(\mathbf p_{\mathrm{stop}}(u)\)为候选动作接入制动后车体最终停车位置，\(\tau\in[0,T_h]\)表示此后停车时间，车辆包围圆与道路/动态目标的有符号最小净空记为\(c(u,\tau)\)，定义

\[
c_{\mathrm{min}}(u)=\min_{\tau\in[0,T_h]}c(u,\tau)
\]
\[
J_{\mathrm{terminal}}(u)=w_t[\max(0,c_{\mathrm{target}}-c_{\mathrm{min}}(u))]^2.
\]

最终在可行候选集合中

\[
u^*=\mathop{\mathrm{argmin}}_{u\in\mathcal U_{\mathrm{strict}}}
  (J_{\mathrm{original}}(u)+J_{\mathrm{terminal}}(u)).
\]

当前预设\`c_target=1.2m\`、\`w_t=3.0\`，属于工程假设，不是独立训练得到的最佳参数。**与不确定性概率安全保证无关，也不能保证长时域决策一定可行。**该代价只是一种可解释的终端安全偏好项，属于已知思路的场景化设计；创新性仍需实验验证。

若没有一个满足严格条件的动作，**退化与braking_hold完全一致**；这避免了第六阶段把原本成功样本变成新的风险冒险样本。

## 公平对照实验

\`\`\`bash
pip install -e '.[dev]'
pytest -q

lidar-eval --scenarios two_targets --stresses nominal,weak_brake \\
 --modes braking_hold,terminal_margin --seeds 3 --max-time 30 \\
 --output results/terminal-margin-ablation --print-episodes --plot

lidar-sim --scenario two_targets --mode terminal_margin \\
 --terminal-margin-weight 3.0 --terminal-margin-target 1.2 \\
 --seed 0 --plot --decision-figure
\`\`\`

输出 \`episodes.csv\`、\`summary_by_scenario_stress_mode.csv\` 和逐种子 \`paired_terminal_margin_effects.csv\`；不得只展示成功/碰撞，须同时展示超时、出界和通行时间。

## 科学与工程限制

1. 停后6秒仅为有限预测窗口，行人未来轨迹仍是理想CV真值，不包含感知噪声、漏检、突然转向等真实扰动。
2. 小车是保守圆形碰撞近似，非精确矩形车辆；时间离散，没有连续时间安全性证明。
3. “终端净空偏好”不等于可证明的递归可行性；如果全部候选都不可行，仍可能发生碰撞或超时。
4. 不应只依据少量随机种子的先导结果称为主要论文创新；需扩充独立场景、增加权重敏感性和保留集评估，特别检查不同道路宽度与目标运动参数。
