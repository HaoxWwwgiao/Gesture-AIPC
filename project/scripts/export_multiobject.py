"""Export static raw YOLO heads with source/weights/class coverage evidence."""
from pathlib import Path
import argparse
import json
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pc.multiobject.yolo import digest, load_package


def export(weights, out, role, training_report=None):
    import torch
    from ultralytics import YOLO, __version__
    out = Path(out)
    if out.exists():
        raise ValueError('Export destination already exists')
    torch.set_num_threads(4)
    model = YOLO(str(weights))
    report = json.loads(Path(training_report).read_bytes()) if training_report else None
    if role == 'cube_detector' and report is None:
        raise ValueError('Cube head needs a training provenance report')
    classes = [model.names[i] for i in range(len(model.names))]
    if report and report['classes'] != classes:
        raise ValueError('Training/export class order mismatch')
    result = model.export(format='onnx', imgsz=640, opset=17, dynamic=False,
                          simplify=False, nms=False, batch=1, device='cpu')
    out.mkdir(parents=True)
    shutil.copyfile(result, out/'model.onnx')
    spec = {'schema': 'gesture.ai_pc.yolov8_package.v1', 'role': role, 'architecture': 'yolov8s',
            'classes': classes, 'trained_classes': report['trained_classes'] if report else classes,
            'input_size_wh': [640, 640], 'input_encoding': 'RGB_FLOAT_0_1_LETTERBOX_114',
            'output_encoding': 'YOLOV8_XYWH_CLASS_PROB_NO_NMS', 'precision': 'fp32',
            'onnx': {'file': 'model.onnx', 'sha256': digest(out/'model.onnx')},
            'source_checkpoint_sha256': digest(weights), 'ultralytics_version': __version__,
            'license': 'Ultralytics AGPL-3.0 / Enterprise; see upstream terms',
            'weight_source': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt',
            'metric_scope': 'BOOTSTRAP_NOT_INDEPENDENT' if report else 'COCO_PRETRAINED_REFERENCE_NOT_CUSTOM_ITEM_VALIDATION',
            'training_report': report, 'npu_validation': 'NOT_RUN', 'physical_execution': False}
    (out/'model.json').write_text(json.dumps(spec, indent=2)+'\n', encoding='utf8')
    load_package(out/'model.json')
    return spec


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--role', choices=['cube_detector', 'object_recognizer'], required=True)
    p.add_argument('--training-report', type=Path)
    a = p.parse_args()
    result = export(a.weights, a.out, a.role, a.training_report)
    print('EXPORTED', a.out, result['onnx']['sha256'])
