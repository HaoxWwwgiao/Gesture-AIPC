# 模型与外部资产

本源码版**不含任何训练参数张量**。`project/artifacts/.../model.json`只保留网络结构、输入输出及训练链元数据，所引用的模型二进制没有随附。

| 原现场当前资产 | SHA256 | 发布状态 |
| --- | --- | --- |
| YOLO六类检测ONNX | `5999c0e1e2d2631cf68b086b1b667ec87eabbb5268b1d6550421371520308cce` | 排除，参数分发覆盖尚未核清 |
| MobileNetV3顶面checkpoint | `9143a31f4ced63410d1bb3b3367139eb48dda68551e0aedce12eceda9b5f8f93` | 排除，参数分发覆盖尚未核清 |
| 原19项冻结清单 | `cf2c959aa9aae7660015a3769ff1edcd6ed4ca8e32dee0f8e3a4fbbb9cbc5778` | 排除；其中还绑定图片/数据等外部资产 |

当前完整视觉入口原路径为`project/config/ai_pc/mode_b_visual_current_20261029.json`，原检测/顶面参数信息在`project/artifacts/multiobject/`下相应`model.json`。配置引用的冻结清单、图片、数据清单、关键点样图和部分相机内外参未随附；缺少时应拒绝运行。

完整运行还需：匹配的相机标定、桌面/基座/TCP变换、CAD网格、实物补偿、抓取及HOME过渡计划、板卡BIT/HWH/XSA、板端固件与工具环境。公开仓库仅有主机代码及部分现场JSON示例，不能按示例坐标驱动另一套机械臂。

许可证核对尚未闭合的范围：Office-Home使用/分发、ImageNet起点参数覆盖、部分照片归属与派生参数的公开方式。LVIS标注CC-BY、TorchVision软件BSD不代替图片/权重授权；NC/ND图片条件也不在这里自动等同于派生模型条款。保留原成功模型作为本地回退，不因源码发布替换或删除。

原始核对记录见`aipc_license_20261103/README_CN.md`及`current_models_final/audit.json`。其中的`redistribution_clearance_complete=false`仍有效；这些历史证据不是本次公开权重的授权书。
