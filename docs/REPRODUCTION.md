# 复现步骤与边界

## 1. 校验源码和冻结模型

```powershell
git clone https://github.com/HaoxWwwgiao/Gesture-AIPC.git
Set-Location Gesture-AIPC
python -B tools/verify_source_release.py
python -B deployment/b_vision/verify_runtime_assets.py --root project
python -B -m unittest discover -s tools -p test_source_release.py -v
python -B -m unittest discover -s deployment/b_vision -p test_runtime_assets.py -v
```

当前版本包含两份参数及独立B视觉配置、标定与投影代码。清单校验只需标准库，不加载模型或设备。新仓库历史不包含原私有开发历史；逐文件来源保存在`SOURCE_PROVENANCE.json`。原源码版离线检查记录仍保留为历史，新增模型运行验证另存`validation/b_visual_runtime_20261103.json`。

## 2. 运行B视觉

按`deployment/b_vision/README_CN.md`分别准备CPU开发或Windows/HIP环境，执行该目录的`run_image.py`。使用者提供一张与标定机位相符的原始图片；源码和参数均已包含，训练数据不参与推理。结果是桌面毫米候选坐标，物理执行为false、路径提交为0。

## 3. 网页离线预览

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-preview.txt
.\.venv\Scripts\python.exe -B project/scripts/run_mode_ab.py --preview --no-browser --out-dir _local/preview_001
```

打开本机网址查看断开设备状态的UI，Ctrl+C退出。该步骤未自动连接相机或板卡。带设备的生产`--run`、`--check-only`仍执行原封存完整包校验；本仓库不包含全部FPGA、历史证据、网格和整机资产。

## 4. 训练和导出

`project/scripts/train_mode_b_public.py`、`train_mode_b_surface.py`和`export_mode_b.py`提供六类检测、顶面训练和静态ONNX导出接口。早期三色代码为`train_multiobject*`。开发CPU版本参考`project/requirements-multiobject-cpu.txt`；数据依赖参考`requirements-mode-b-data.txt`。先执行各脚本的`--help`，实际训练需自行取得适用的数据、标签及初始化文件；历史脚本可能还为训练报告核验原始起点SHA。本仓库没有公开训练数据，也未声称可只凭现有文件重训出同SHA参数。

## 5. 规划与整机

源码保留主机规划、状态机和板卡接口。当前规划会读取URDF和碰撞STL，因此完整规划还需要匹配网格；纯视觉入口不需要。整机还需匹配板端固件、计划、HOME过渡和现场标定。原完整包门禁未删改，不能把独立视觉文件完整解释为整机抓放已经验收。

当前公开下载确认见`WEIGHT_DISTRIBUTION_RECORD.json`，第三方来源声明见根目录`THIRD_PARTY_NOTICES.md`。旧审计文件保持原日期和原结论，当前发布决定单独记录。
