# 源码版复现边界与步骤

## 1. 取得和验证源码

```powershell
git clone https://github.com/HaoxWwwgiao/Gesture-AIPC.git
Set-Location Gesture-AIPC
python -B tools/verify_source_release.py
```

本仓库是新建历史的源码快照；`docs/SOURCE_PROVENANCE.json`记录原始文件SHA，`source_manifest.json`记录本次发布全部源码。原 `project/mode_ab_manifest.json` 校验的是完整封存运行包；它不用于证明这个子集可运行，也没有为适应缺文件而放宽。

源码版另附开发目录中的4个团队原创数值计划，作为边界显示/规划的参考，逐项标注`DEVELOPMENT_NUMERIC_PLAN_REFERENCE_SUPPLEMENT`；它们不包含网格或固件，不能替代完整部署包。源码版`pc/mode_ab/http.py`增加了有界丢弃未授权POST请求体的处理，避免Windows上返回403前因未读请求体触发TCP重置；不解析该请求或排队动作。该差异记录在来源清单，未改私有生产目录。

## 2. 运行离线测试与网页预览

先按根README执行标准库测试。可选创建独立Python3.12虚拟环境并安装`requirements-preview.txt`，再运行：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s project/scripts -p test_mode_ab_http.py -v
.\.venv\Scripts\python.exe -B -m unittest discover -s project/scripts -p test_mode_ab_diagnostics.py -v
.\.venv\Scripts\python.exe -B project/scripts/run_mode_ab.py --preview --no-browser --out-dir _local/preview_001
```

HTTP测试使用假相机/假板状态及本机回环连接，不访问真实设备。Node.js可选执行`node project/scripts/test_mode_ab_ui.cjs`。不运行`test_*.py`全目录发现：部分历史测试依赖未包含的运行包、网格或板卡。

## 3. 查看训练和导出接口

`project/scripts/train_mode_b_public.py`为六类检测训练；`train_mode_b_surface.py`为顶面训练；`export_mode_b.py`为静态ONNX导出。`train_multiobject*`为早期三色路线。训练脚本来自开发工作区，逐文件标注`DEVELOPMENT_TRAINING_SOURCE_SUPPLEMENT`，没有冒充封存生产代码。

开发CPU环境历史版本见`project/requirements-multiobject-cpu.txt`，数据依赖见`project/requirements-mode-b-data.txt`；请与生产AMD环境分开。依赖安装可能需对应上游版本可用的索引，源码发布未验证全新联网安装历史环境。

先查`python project/scripts/train_mode_b_public.py --help`、`python project/scripts/train_mode_b_surface.py --help`及`python project/scripts/export_mode_b.py --help`。训练实际需要自行合法取得数据、标签及初始化/暖启动参数。某些历史脚本即使使用暖启动，报告阶段仍校验原始起点文件SHA，不能只补一个当前checkpoint就假定依赖齐全。

没有提供一键下载全部历史素材的脚本，也没有声称可仅凭本仓库重训出与现场同精度、同SHA的模型。公开数据来源和参数分发条款尚未全部核清；不要从旧权重继续微调后宣称历史来源已经清除。

## 4. 完整推理、定位和抓放

当前权重身份、路径及外部依赖见`MODEL_ASSETS.md`。相机移动、变焦、分辨率变化后必须重新标定；配置文件中的旧桌面坐标、补偿和实际尺寸仅为原现场记录。完成模型/设备/标定验证后，仍需配套板端运行包、路径计划和真实硬件联测。此源码版不包含它们，不能从网页预览直接完成抓放。

本次验证限定为代码语法、指定离线回归和短时无设备网页服务；没有重新验证识别精度、毫米精度、NPU、HIP或物理成功率。具体运行结果见`validation/source_checks.json`。
