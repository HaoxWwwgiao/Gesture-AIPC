# Gesture-AIPC：AI PC识别、定位与规划

手势驱动桌面递送项目的AI PC部分，团队原创源码采用 **AGPL-3.0-only**。包含六类目标检测、神经网络顶面分割、毫米坐标计算、主机端规划和网页控制代码。

**当前已包含B视觉所需的两份冻结权重、模型配置与标定参考。** 项目负责人于2026-11-03明确确认“权重都已允许下载”，发布身份与SHA见[发布记录](docs/WEIGHT_DISTRIBUTION_RECORD.json)。训练数据集、照片、FPGA实现、CAD网格及原私有仓库历史不随附。原成功权重字节保持不变。

## 获取两份权重

正常克隆本仓库即可获得；两份文件均直接存储于Git，未使用Git LFS。

| 文件 | 完整SHA256 |
| --- | --- |
| [检测model.onnx](project/model_package/mode_b/detector_background_v2/model.onnx) | `5999c0e1e2d2631cf68b086b1b667ec87eabbb5268b1d6550421371520308cce` |
| [顶面weights.pt](project/model_package/mode_b/surface_eraser_repair_v1/weights.pt) | `9143a31f4ced63410d1bb3b3367139eb48dda68551e0aedce12eceda9b5f8f93` |

已有整机包的接收方，将这两份文件按相同`project/model_package/...`路径放入，保留原`model.json`，再执行原视觉冻结检查。仅下载页面HTML不能替代二进制文件；直链见[模型说明](docs/MODEL_ASSETS.md)。

## 先检查文件完整性

Python3.12，在仓库根执行；只需标准库，不加载模型或访问设备：

```powershell
python -B tools/verify_source_release.py
python -B deployment/b_vision/verify_runtime_assets.py --root project
```

预期`release_integrity_passed=true`、`b_visual_runtime_assets_complete=true`及31项运行文件全部`MATCH`。`runtime_ready=false`指整机外部硬件/网格资产未随附，不表示B视觉缺权重。

## 保存图像识别与毫米坐标

独立视觉入口不依赖训练集、机械臂、FPGA或CAD。按[部署步骤](deployment/b_vision/README_CN.md)准备CPU或Windows/HIP环境，在当前标定机位的一张1920×1080原始图上执行：

```powershell
python -B deployment/b_vision/run_image.py --root project --image C:\path\to\raw.png --same-camera-pose --backend onnx_cpu --out _local/vision_001
```

AMD环境显式改为`--backend rocm`，拒绝CPU回退。输出六类检测及三类可抓取物品的`desk_global_mm`候选坐标；不生成机械臂指令。相机位置改变后需重新标定，不能把旧坐标变换套到任意图片。

## 功能与验证范围

- 六类为红/黄/蓝木块、橡皮、短圆柱药瓶、剪刀；B模式可抓取目标红块/橡皮/药瓶。A模式保留三色固定槽位的主机配置接口。
- YOLOv8s检测；MobileNetV3-Small编码器及项目顶面解码器；轮廓经内外参和已知物体高度投影到毫米平面。
- 已在独立目录对一张保存红块图完成CPU识别及坐标回放，一致性记录见`validation/b_visual_runtime_20261103.json`。这不是新场景毫米绝对精度验收。
- 当前生产架构为Windows AMD ROCm/HIP。此次没有重新验证HIP、NPU或整机；NPU仍未验收。
- 网页离线预览、训练/导出源码、规划源码均保留；完整抓放仍需匹配的规划网格、板端运行包和现场校验。原完整包门禁保持。

详细说明：[复现步骤](docs/REPRODUCTION.md) · [B视觉部署](deployment/b_vision/README_CN.md) · [第三方声明](THIRD_PARTY_NOTICES.md) · [逐文件来源](docs/SOURCE_PROVENANCE.json)。
