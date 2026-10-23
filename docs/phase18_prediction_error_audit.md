# H1 预研究：独立轨迹标签下的LiDAR短时位置预测误差审计

> **当前状态**：只增加可复用的评价程序、数据契约和合成回归测试。尚无实际雷达轨迹真值、实际观测误差统计，也没有完成真实H1检验。不修改任何安全规划判据或底盘命令。

## 研究目标和边界

原始课题调研报告提出H1：在相同预测时域内，观测质量不同（例如点数稀少、观测陈旧）时预测误差可能不同；在相近覆盖水平下，质量分组的安全占据包络可能比统一余量更紧凑。**这个假设必须由未参与校准的独立场景数据判断，不允许假设它天然成立。**

当前模块为 `src/lidar_sim/track_prediction_audit.py`，仅对已存在的目标估计计算CV未来位置，与单独记录的独立标签对照。在误差评价之后才访问标签，因此**参考位置绝不回流到在线预测**。

### 输入1：每次决策时的估计快照

与PR #20完全一致的 `JSONL`，包括固定 `odom` 坐标系、`t_s`、`ego`、`actuator`、`road`、`tracks`。每个目标标明 `id`、`observed_at_s`、`x_m`、`y_m`、`vx_mps`、`vy_mps`、`radius_m`、可选 `point_count`。坐标属于**观察当时**，时间归一成同一时钟；不会因为错过一帧自动构造完美真值。

从PR #21起，`MovingObstacle`增加默认0秒的 `reference_time_s` 锚点；历史合成场景以0为锚点，行为完全不变；真实观测在 `observed_at_s` 锚定运动状态，避免把epoch量级的时间乘速度再相减。

### 输入2：完全独立的标签文件（CSV）

必须由与被测CV-KF估计**独立**的标注/高精参考测量生成，例如经时钟和地面单应矩阵校准的视频轨迹，或外部定位设备。CSV字段如下：

~~~csv
frame,track_id,time_s,x_m,y_m
odom,track_7,12.9,13.12,-3.58
odom,track_7,13.4,13.03,-3.41
odom,track_7,14.4,12.82,-3.07
~~~

其中 `track_id` 与估计ID经过人工核对匹配，错误对应应记录并剔除为不可评分片段。**不能用同一个KF输出的未来后验当作独立真值**。标签的 `reference_kind=independent_annotation` 只是来源声明，脚本无法自动验证真实独立性。标签与预测点的时间差默认不超过0.05秒；超出时记为缺失，**绝不插值补齐，更不能记零误差**。

### 输入3：按独立采集场景隔离的清单

`manifest.csv`：每一行对应一个完整独立采集episode；源路径相对于清单位置解析：

~~~csv
episode_id,split,snapshot_jsonl,reference_csv,reference_kind
walk01,calibration,walk01/tracks.jsonl,walk01/independent_labels.csv,independent_annotation
walk02,calibration,walk02/tracks.jsonl,walk02/independent_labels.csv,independent_annotation
walk13,test,walk13/tracks.jsonl,walk13/independent_labels.csv,independent_annotation
~~~

同一录像/同一轨迹的相邻帧**不能**被拆分到两个不同episode然后分别放校准集与测试集。代码拒绝重复episode ID或复用同一个数据文件，但人为复制/重命名同一素材无法识别，必须由数据管理保证。开发过程冻结分组阈值，之后不凭测试集结果反复修改。

## 评价单位和误差定义

默认未来三个时刻 `h∈{0.5,1.0,2.0}s`。每个有效估计起点 `t`，预测 `p_hat(t+h)`；标签 `p_ref(t+h)`。定义该预测起点的整段最差位置误差：

`S_origin = max_h ||p_hat(t+h)-p_ref(t+h)||_2`。

需要全部三个标签才能计为**完整整段**。漏标、过期目标和不完整段计入 `incomplete_or_stale_segments`，不能擅自当作成功。仍须报告缺失率，因为只筛可评分帧可能导致明显的选择偏差。

由于10Hz连续预测高度相关，校准/测试样本单位是**每个episode在同一观测质量组内的最大完整整段误差**，即 `S_episode,group = max_origin S_origin`。从而避免把同一段场景的数百帧当作几百个独立样本。相同episode可能同时包含不同组，但每组各最多贡献一个独立episode单元。

分组根据预测时可观测的字段预先固定：
- `good_observation`：当前轨迹点数不少于10，并且当前观测龄期≤0.15s；
- `degraded_or_unknown`：其余样本（含点数未知或陈旧，但已经被大于0.35s时效上限排除的目标不参与校准）。

**这只是固定的第一版诊断分组，不代表真实观测质量界限已被校准，也没有将跟踪器自身的σ当成可靠概率置信度。** 真实数据如显示距离是主要混杂因素，需在开发集另行研究距离分组并在独立测试集冻结检验。

## 内部比较：pooled vs grouped

使用 `alpha=0.10` 的分位排序 `k=ceil((n+1)*(1-alpha))`；当 `k>n` 时不给有限上界，输出 `insufficient_independent_calibration_episodes`。例如90%目标下，一个分组少于9个独立完整episode不能输出有限分位半径。

- **pooled**：每个校准episode取全部组的最差误差，建立同一个全局半径；
- **grouped**：分别按该组episode最差误差建立各自半径；
- **test**：只计算事先隔离的测试episode单元的实际覆盖比例，分别展示半径、分组数据量、未能评分episode数量。

这些数值是**在所给参考数据和交换性等条件下的经验整段误差覆盖**。标注误差、非交换场景、跨目标关联失败、重度漏检及在线多次重规划都会削弱推广。不能把90%片段覆盖写成90%任务安全率，也不能把这种基本统计分位校准包装成原创共形算法。当前只比较位置误差半径，**尚未转换为带障碍几何的占据包络，也不影响规划器**。

## 执行方法

~~~bash
python -m pytest -q tests/test_track_prediction_audit.py tests/test_tracked_shadow.py
python -m lidar_sim.track_prediction_audit \
  --manifest /path/to/manifest.csv \
  --output /path/to/audit-results \
  --horizons 0.5,1.0,2.0 \
  --reference-tolerance-s 0.05 \
  --alpha 0.10
~~~

生成：
- `episode_segment_error.csv`：每段完整/不完整数、每个episode/group最差误差、校准/测试标识；
- `coverage_audit.csv`：pooled、grouped半径，独立episode计数，留出测试覆盖和缺失率；
- `run_config.json`：冻结的阈值、预测时域、样本划分与来源声明。

测试自建12段校准/4段测试**合成数据**，专门验证阈值计算、样本不足拒绝、未来标签缺失不当作零误差、帧/时戳校验、数据集泄漏检测、epoch尺度观测时间数值稳定。这些测试结果**不能作为真实雷达跟踪性能、实际预测包络改进或研究假设H1成立的证据**。

## 下一阶段的真实数据验收门槛

1. 导入真实记录：目标估计（含点数/距离/ID/漏检情况）、带时间戳的odom，及与该跟踪器独立的参考标注；校核TF和时间基准。
2. 按**整段独立采集episode**划分校准/测试。至少检查每组是否达到输出有限统计阈值的样本量，并报告失踪/漏检片段比例。
3. 定量检验H1：同等覆盖目标下分组半径/面积是否减小；性能差异需要在独立场景中复核。若分组没有稳定收益，及时取消“分组校准”作为独立创新。
4. 确立参考数据质量后，再将已校准占据误差加入有限轨迹的动态碰撞筛选，做固定余量、统一半径、分组半径消融；不能在尚未验证覆盖前直接把σ乘一个数当作可信安全约束。
