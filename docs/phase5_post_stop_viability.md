# 第五阶段：诊断停车碰撞并评估后停驻时间的时空占据

## 失败回合实际复现

[GitHub 失败轨迹诊断（日志与CSV）](https://github.com/cydonia0717-crypto/lidar/actions/runs/39191707117)

双目标失败的共同问题是：约10—12秒，小车已经无可行候选速度/转角，进入 emergency_brake；数秒后，骑行者经过小车停止位置造成碰撞。尤其 \`braking\` 既未在早期拒绝有关候选，也没有处理停止后的目标运动。

不能据此证明其它交通场景也有同样故障。针对逐帧记录直接进行可复核的消融。

## 消融设计：braking vs braking_hold

- 旧：延迟已提交指令→候选执行0.40s→遵守最大减速度和已达到前轮转角的制动尾迹。
- 新：保留原三段轨迹，在其**最后一个停止状态**追加固定位置、速度0的有限时间段（默认6.0s）。
- 新检查继续用动态目标在**相应未来时刻**的预测位置和道路宽度约束判断停止区是否被侵入。
- 两种方法使用相同的CV目标真值、预测误差附加半径、车辆动力学、道路环境和评分函数，因此两者差异严格来自停车后占据检查。
- 可选参数：\`--stationary-hold 6.0\`。零秒用于单元回归：\`braking_hold\` 必须和 \`braking\` 返回同样的可行候选。
- 新增明确的单元测试：假设小车0.6秒刹停，但行人5秒之后经过小车停车点。旧模式无法识别该风险，新模式可以检测并拒绝相应候选。

**科学边界：**
1. 这只是有限6秒内的停车位置检查，不意味着真实行人未来永远不改变方向，也不提供无限时间安全性保证。
2. 安全检查依然基于离散采样、理想CV目标运动和保守圆形车辆几何，未使用真实雷达。
3. 有限候选集可能仍然全部不满足约束，此时自动退化策略仍可能失败。更强的停车后约束甚至可能导致保守性增加、成功率下降或过早退化。
4. 尽管该模块能通过人工构造的边界单元测试，**论文性能结论必须以独立场景配对结果为准**。

## 实际命令

\`\`\`bash
pip install -e '.[dev]'
python -m pytest -q

lidar-eval --scenarios two_targets --stresses nominal,weak_brake \\
    --modes braking,braking_hold --seeds 3 --max-time 30 \\
    --output results/post-stop-ablation --plot

lidar-sim --scenario two_targets --mode braking_hold --stationary-hold 6 \\
    --seed 0 --plot --decision-figure
\`\`\`

完整CI将自动上传逐回合日志、配对结果与图到 GitHub Actions artifact。对于原有双目标失败种子，应尤其检查 braking_hold 能否防止“先停后撞”，而不能用预期效果图冒充实际改进。
