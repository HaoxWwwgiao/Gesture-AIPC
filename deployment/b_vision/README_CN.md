# B视觉：原始冻结权重的独立推理入口

当前仓库已提供两份参数及31项运行依赖。此入口执行保存图片→六类检测→三类顶面→桌面毫米候选坐标，不访问相机、串口、网络机械臂或FPGA。

## 1. 获取与检查

```powershell
git clone https://github.com/HaoxWwwgiao/Gesture-AIPC.git
Set-Location Gesture-AIPC
python -B deployment/b_vision/verify_runtime_assets.py --root project
```

全部31项应为`MATCH`。已有整机包的队友只需补缺失的两份同SHA权重到原路径，再运行该包原有核验；本仓库不替换整机板端文件。

## 2. CPU开发环境

Python3.12。历史CPU版本参考`project/requirements-multiobject-cpu.txt`，与AMD环境分开安装：

```powershell
py -3.12 -m venv .venv-cpu
.\.venv-cpu\Scripts\python.exe -m pip install -r project/requirements-multiobject-cpu.txt
.\.venv-cpu\Scripts\python.exe -B deployment/b_vision/run_image.py --root project --image C:\path\to\raw.png --same-camera-pose --backend onnx_cpu --out _local/vision_cpu_001
```

依赖版本的全新联网安装本轮未重跑；本轮使用已具备相应CPU依赖的环境完成同图推理。若上游历史版本不可用，保留安装错误再调整环境。

## 3. Windows AMD/HIP环境

原生产目标为gfx1151、Python3.12、torch2.13.0+rocm10.0.0、torchvision0.28.0+rocm10.0.0、onnx2torch1.5.15。完整锁见`project/requirements_mode_ab_lock.txt`。准备匹配驱动后：

```powershell
powershell -ExecutionPolicy Bypass -File project/setup_mode_ab.ps1 -Python C:\path\to\Python312\python.exe
.\project\.venv-ab-rocm\Scripts\python.exe -B deployment/b_vision/run_image.py --root project --image C:\path\to\raw.png --same-camera-pose --backend rocm --out _local/vision_hip_001
```

原setup会联网安装依赖并做真实HIP设备算术探测；此轮没有执行它。不要加`-VerifyModels`运行旧全包数值验证，因为它另需历史样本。`rocm`后端拒绝CPU回退；SDK包版本与实际HIP运行时版本分别记录。NPU并未因此得到验收。

## 4. 图像和结果边界

输入需为当前标定机位的1920×1080原始图片。`--same-camera-pose`是调用者声明，不会自动重新标定。换机位或分辨率后先更新标定，不使用旧毫米变换。输出位于新的`--out`目录，已有目录会拒绝覆盖。

`result.json`包含六类识别，以及红块、橡皮和药瓶的顶面与物体中心候选。三者半高分别15、7.5、23 mm，坐标系为`desk_global_mm`。没检出、多个实例、几何不合格时返回拒绝/缺失状态，不伪造坐标。没有机械臂基座运动或抓放命令。

## 5. 验证

```powershell
python -B -m unittest discover -s deployment/b_vision -p test_runtime_assets.py -v
```

本轮使用原保存红块图完成CPU识别和毫米结果回放，与该同帧历史参考一致；这不等同于独立毫米绝对精度、六类全覆盖、HIP或整机验收。记录见`validation/b_visual_runtime_20261103.json`。训练图片与示例照片未随此次上传。
