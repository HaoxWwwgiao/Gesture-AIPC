# 当前模型与运行资产

项目负责人于2026-11-03确认“权重都已允许下载”，并明确要求上传所需文件。当前版本已发布两份原始冻结权重和31项独立B视觉运行文件；发布依据保存在`WEIGHT_DISTRIBUTION_RECORD.json`，原始模型SHA、配置和来源声明保持。

| 资产 | SHA256 | 字节数 |
| --- | --- | --- |
| `project/model_package/mode_b/detector_background_v2/model.onnx` | `5999c0e1e2d2631cf68b086b1b667ec87eabbb5268b1d6550421371520308cce` | 44666590 |
| `project/model_package/mode_b/surface_eraser_repair_v1/weights.pt` | `9143a31f4ced63410d1bb3b3367139eb48dda68551e0aedce12eceda9b5f8f93` | 6148747 |

直接下载：

- [检测ONNX](https://raw.githubusercontent.com/HaoxWwwgiao/Gesture-AIPC/main/project/model_package/mode_b/detector_background_v2/model.onnx)
- [顶面checkpoint](https://raw.githubusercontent.com/HaoxWwwgiao/Gesture-AIPC/main/project/model_package/mode_b/surface_eraser_repair_v1/weights.pt)

对应`model.json`已包含在同目录。接收方按原相对路径放置，不重新导出或替换元数据来凑SHA。独立视觉运行清单为`deployment/b_vision/MINIMUM_VISION_RUNTIME.json`，核验命令见该目录README。

训练图片、标签全集、历史包不需要用于本独立推理入口，也未上传。旧19项冻结门禁和整机封存清单保持原样；它们绑定的历史文件范围大于独立视觉入口。此入口另列实际推理依赖，没有静默放宽原门禁。

相机内参、桌面变换、工作区和尺寸配置是原现场记录。相机移动、变焦或分辨率变化后需重做相应标定；不能直接用这些坐标驱动另一套机械臂。纯视觉不需要CAD，现有规划入口仍读取URDF及碰撞STL；完整规划/抓放所需外部资产见复现说明。

历史许可审计保留在`aipc_license_20261103/`。其中的原始待确认标记反映当时的核对状态；本次公开下载依据单独记录为项目负责人确认，不改写历史证据或声称取得了本轮未收到的上游书面答复。
