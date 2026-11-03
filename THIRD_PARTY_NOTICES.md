# AI PC 第三方声明

团队有权授权的源码按[AGPL-3.0-only](LICENSE)提供；第三方原有声明优先。本仓库没有打包依赖运行库安装器、模型参数或训练照片。

| 来源 | 使用和条款 |
| --- | --- |
| [Ultralytics YOLOv8s](https://github.com/ultralytics/ultralytics/tree/v8.4.89) | 检测、训练及导出；采用AGPL路线，未声称Enterprise许可。[对应上游许可全文](docs/open_source_audit_20261103/upstream/ultralytics-version-license.txt)。 |
| [TorchVision](https://github.com/pytorch/vision) | MobileNetV3-Small编码器；[软件BSD-3-Clause](docs/open_source_audit_20261103/upstream/vision-license.txt)。预训练权重与数据条款另行核实；不是SWAG路线。 |
| [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html) | 历史训练链使用；作者声明非商业研究/教学fair use，不是任意再分发许可。本库不提供照片或模型参数。 |
| [LVIS](https://www.lvisdataset.org/dataset) | 标注CC-BY-4.0，API代码BSD；[API声明](docs/open_source_audit_20261103/upstream/lvis-license-master.txt)。照片来自COCO/Flickr，逐图许可单独核对。 |
| [COCO](https://cocodataset.org/#termsofuse) | [API代码BSD](docs/open_source_audit_20261103/upstream/coco-license-data.txt)与照片各自许可分开，图片未随附。 |
| NumPy、SciPy、OpenCV、PyTorch、ONNX/ONNX Runtime等 | 通过依赖管理器单独安装，保留各包自身许可；历史依赖元数据见审计目录，不等同于全环境SBOM认证。AMD SDK和工具依其条款，不随附安装器。 |

详细历史审计见[AI PC核对记录](docs/aipc_license_20261103/README_CN.md)。该记录先于本源码版，提到的私有仓库、目录和素材不是本次公开文件清单。参数SHA和待补事项见[外部资产](docs/MODEL_ASSETS.md)。

如通过网络提供此AGPL程序的修改版本服务，应按AGPL要求向远程用户提供相应源码；提供本仓库链接时须确保它对应实际运行版本及修改。本次源码发布并不宣称已满足尚未发布资产的全部权利与分发条件。
