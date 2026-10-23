# H1 补充：独立可见性标注与目标漏跟踪审计（只读）

> **定位：** 本模块解决已跟踪样本上的预测误差评价具有严重选择偏差的问题。**仅有跟踪器输出，无法判断未输出的目标是否存在。** 本迭代新增对独立可见目标的几何关联核验，仍未声称检测召回率、目标关联准确率或整车安全性能已达到任何标准。

## 与现有两条研究管线的关系

1. PR #20 `tracked_shadow.py`：接收跟踪估计，安全地构造规划器输入。没有跟踪结果时**不能推断世界无障碍**。
2. PR #21 `track_prediction_audit.py`：对已经报告出来的轨迹，使用未来独立标注评估CV预测误差。完整回合缺标不得计为已覆盖。
3. 本PR `independent_visibility_audit.py`：在独立相机/人员标注确认目标处于LiDAR有效视野时，检验同一时刻是否有**可唯一几何对应的**目标跟踪记录；绝不依赖“只能在跟踪成功后才生成的标签”。

三者都离线执行，不发送速度/转角命令，不修改已验收的第15—17阶段规划器。

## 标注来源：另一套传感器或人工独立校验

每个 `visibility_csv` 必须包含：

~~~csv
time_s,frame,object_id,visibility,x_m,y_m
12.5,odom,person_from_camera_A,visible,13.1,-3.8
12.5,odom,person_from_camera_B,visible,14.2,-2.6
12.5,odom,person_behind_occluder,occluded,,
12.6,odom,person_A,unknown,,
~~~

- `object_id` 是外部参考来源的**物体编号**，不应直接复制跟踪器分配的`track_id`来证明算法本身的关联正确。
- `visibility` 仅允许 `visible`、`occluded`、`out_of_fov`、`unknown`。只有经独立标注为 `visible` 的行可进入分母。**标注人员必须事先定义LiDAR视野、遮挡与最低可观测条件，软件无法验证标注是否真实独立。**
- `x_m/y_m` 在 `visible` 时必须是与跟踪输入同一 `odom` 坐标的有限数值；其他三种状态可以留空。不支持直接混用像素坐标、GPS/RTK经纬度和 LiDAR `base_link` 点位。
- 时钟统一，默认在±0.05秒内找最近的跟踪快照；若没有快照、恰好与两帧等距则记为不可评分，不在背后插值或创造观测。
- 每个在时间上有效的跟踪轨迹按原始`observed_at_s`和CV速度预测到**参考标注的时刻**。在事先设定的 `--gate-m`（默认0.75m）内作一对一几何候选匹配。
- 若没有可匹配轨迹，记`visible_but_no_gated_track`；如多个轨迹均符合，或一个轨迹同时可能对应多个独立可见目标，记`ambiguous_*`，**不得强行列为成功跟踪**。
- 未见轨迹、漏记整帧、含义模糊时提供**保守/乐观的关联事件比例**作为诊断敏感性区间，不把它宣传为概率置信区间。被标记为`occluded/out_of_fov/unknown`的目标既不应计成功，也不应擅自计为跟踪漏检。

## 清单与防止数据集泄漏

使用独立清单，要求同一完整采集session的多段episode全部处于同一划分：

~~~csv
episode_id,session_id,split,snapshot_jsonl,visibility_csv,annotation_kind
walk_cal1,morning_walk_01,calibration,cal1/tracks.jsonl,cal1/visibility.csv,independent_visibility
walk_test1,evening_walk_03,test,test1/tracks.jsonl,test1/visibility.csv,independent_visibility
~~~

程序明确拒绝同一个`session_id`跨校准/测试、重复episode编号以及多个episode复用同一输入或标注文件。人工复制并重新命名相同素材仍然无法自动识别，必须由采集人员维护 provenance。这里的`calibration/test`只是与PR #21分组划分统一，**本程序目前不拟合任何检测模型或阈值**：所有匹配门限需先在开发集确定，并在测试集冻结。

## 运行

~~~bash
python -m lidar_sim.independent_visibility_audit \
  --manifest tests/fixtures/visibility_audit/manifest.csv \
  --output results/visibility-synthetic \
  --gate-m 0.75 --time-tolerance-s 0.05 \
  --max-track-age-s 0.35
~~~

提供两个文件：
- `visibility_events.csv`：每个外部标注可见/非可见目标事件的状态、匹配轨迹、时间与关联误差；
- `visibility_episode_summary.csv`：每回合独立可见标签数量、几何唯一关联、可见但未匹配、歧义或无记录帧，及用极端未知假设得到的关联率上下界；
- `run_config.json`：实验门限和输入来源声明（不证明标注独立）。

`tests/fixtures/visibility_audit` 是完全手工构造的**合成示例**：可见目标也可以被跟踪器漏掉，另有遮挡和无记录帧。它只用于软件正确性，不是雷达实际性能结果。

## 论文中可以写与不能写

**可以写：**“建立了以独立可见性标注为参考的漏跟踪审计接口，使用固定时空门限验证报告轨迹与可见目标的几何关联，单独报告缺帧、歧义、可见未跟踪等数据质量事件。”

**不能写：**“实际LiDAR目标召回率达到95%”“算法对遮挡目标已保证安全”“独立标签的轨迹关联完全正确”“未检测到目标代表环境安全”。真正的检测召回率需要先确认外部参考对象确实处于传感器可靠探测区、存在定义明确的独立对象关联真值和经过校准的观测时序。此软件评估不提供这些证据。

实际ROS bag的读取和话题盘点已由PR #21提供 `rosbag_inventory.py`；拿到具体topic/消息schema后，才能设计有证据依据的出口，而不是编造实时接入的完成度。
