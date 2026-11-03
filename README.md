# Gesture-AIPC：AI PC 端源码

手势驱动桌面递送项目的 AI PC 部分，采用 **AGPL-3.0-only**。此仓库独立发布源码，保留六类物体检测、神经网络顶面分割、毫米坐标计算、主机端路径规划和网页控制代码。

**这是源码版，不是完整可部署运行包。** 未包含模型权重、照片、训练数据、FPGA 工程/固件、CAD 网格和原私有仓库历史。原有成功模型没有替换；本次发布没有重新训练或做实体检验。

## 功能与范围

- B 模式六类：红木块、橡皮、短圆柱药瓶、剪刀、蓝木块、黄木块；可抓取目标为前三类。药瓶直径25 mm、高46 mm，橡皮65×15×15 mm是当前现场配置。
- A 模式保留红/黄/蓝固定槽位的主机配置接口；手势识别 FPGA 实现属于外部模块。
- 检测使用 YOLOv8s；顶面使用 MobileNetV3-Small 编码器及项目自建二分类解码器。物体顶面轮廓经相机内外参、已知物体高度转换为桌面毫米坐标，再用于主机端规划；不是仅用检测框中心代替抓取中心。
- 当前封存生产路线为 Windows AMD ROCm/HIP；本次 CPU 离线检查不能证明 NPU 或 AMD 目标机验收。NPU 尚未验收。

## 先执行无硬件检查

Python 3.12，在仓库根目录运行；此步只需标准库：

```powershell
python -B tools/verify_source_release.py
python -B -m unittest discover -s tools -p test_source_release.py -v
python -B -m unittest discover -s tools/licensing -p test_aipc_sources.py -v
python -B -m unittest discover -s project/scripts -p test_mode_ab_control.py -v
```

返回 `source_subset_passed=true` 仅表示源码文件清单检查通过，`runtime_ready=false` 是预期结果。

可选网页离线预览（只启动本机服务，不点击连接相机等设备操作）：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-preview.txt
.\.venv\Scripts\python.exe -B project/scripts/run_mode_ab.py --preview --no-browser --out-dir _local/preview_001
```

打开终端输出的本机网址；使用 Ctrl+C 退出。输出目录须是新的。预览中设备处于未连接状态，不能执行抓放；生产 `--run`、`--check-only` 的原始完整包校验保持生效。

## 目录

| 路径 | 内容 |
| --- | --- |
| `project/pc/mode_b/`、`project/pc/multiobject/` | 六类检测、顶面、定位和推理后端 |
| `project/pc/mode_ab/`、`project/web/mode-ab/` | 生产网页及状态机/规划接口 |
| `project/scripts/train_*`、`export_*` | 开发训练、导出源码补充 |
| `project/ps/robot/` | 主机协议与寄存器编码 Python 依赖，不含 PL 实现 |
| `project/config/ai_pc/` | 现场设备配置示例，不适用于任意相机/桌面 |
| `docs/SOURCE_PROVENANCE.json` | 原封存源码/开发训练补充的逐文件来源与SHA |
| `source_manifest.json` | 此源码版完整文件清单 |

运行边界及训练入口见 [复现说明](docs/REPRODUCTION.md)，外部资产见 [模型与标定说明](docs/MODEL_ASSETS.md)，许可见 [第三方声明](THIRD_PARTY_NOTICES.md)。审计中的历史文件路径、哈希和实测描述不表示这些资产包含于本仓库。
