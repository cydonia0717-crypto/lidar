# 第四阶段先导仿真实验报告（自动化实际运行）

来源：GitHub Actions 成功完成的工作流
https://github.com/cydonia0717-crypto/lidar/actions/runs/39189881498

运行设置：
- 2类场景：crossing、two_targets
- 2类压力：nominal、weak_brake
- 3个种子：0、1、2
- 3个算法：predictive、delay_aware、braking
- 单次仿真上限30秒，总共**36次实际数值仿真**。
- 三种方法在同一 case_id 下使用同一世界几何、动力学参数与执行延迟；都使用理想CV障碍物真值，尚无雷达观测噪声。
- 实验算法未经过单独按种子调参。该试验仅用于发现问题；样本量太小，不能宣称统计显著性。

## GitHub Actions 日志中可直接核验的结果

| 场景/压力 | predictive 成功/碰撞 | delay_aware 成功/碰撞 | braking 成功/碰撞 |
|---|---|---|---|
| crossing / nominal | 3/3；0/3 | 3/3；0/3 | 3/3；0/3 |
| crossing / weak_brake | 3/3；0/3 | 3/3；0/3 | 3/3；0/3 |
| two_targets / nominal | 2/3；1/3 | 1/3；2/3 | 1/3；2/3 |
| two_targets / weak_brake | 2/3；1/3 | 1/3；2/3 | 1/3；2/3 |

单位为“成功次数/3；碰撞次数/3”。失败中此处均为碰撞，但不能排除其他场景中的超时或出界。

**关键结论：** 在这批样本里，**braking 与 delay_aware 的成功/碰撞数量完全相同，没有观察到增加制动筛选带来的通过率提升**。尤其在双目标场景里，predictive 反而有更好的结果；不能借此宣称制动过滤普遍更安全，也不能把3个种子的比率外推到真实机器人。下一步必须定位失败动作的时间序列与后备筛选原因，而不是挑选更容易成功的场景。

## 紧急补充：保存失败回合的完整决策日志

研究程序新加 `--failure-traces-per-group`（默认2），将每个场景×压力×模式最多两次碰撞、出界或超时的原始状态和决策过程写到 `failure_traces/`。包含自车时刻、实际执行与发出指令、速度、转角、最小净空、候选后备拒绝数量、未验证后备状态及评价函数分量。

下一步对双目标失败回合应逐帧确定：
1. 实际碰撞时刻与发生碰撞的目标；
2. 首次进入危险时间窗口的时刻、自车速度和前轮转角；
3. 当时是否有一条符合制动约束的安全候选轨迹；
4. 控制延迟前缀是否限制了撤回动作的能力；
5. 是否陷入 `nominal_no_backup` 或 `emergency_brake` 退化；
6. 道路边界/轨迹代价是否造成车辆犹豫或错误变道。

不能在没有逐帧日志之前把“失败原因”归结为某个算法缺陷。

## 新增的可复现调试命令

```bash
lidar-eval --scenarios crossing,two_targets \
 --stresses nominal,weak_brake \
 --modes predictive,delay_aware,braking --seeds 3 \
 --max-time 30 --failure-traces-per-group 2 \
 --output results/pilot_with_traces --plot
```

结果全部是**人工合成的二维运动学仿真**，并不是实车采集或毕业论文已经得到的改进结论。
