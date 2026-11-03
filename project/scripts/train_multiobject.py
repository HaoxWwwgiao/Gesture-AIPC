"""Fine tune YOLOv8s; hue changes disabled because colour selects the task."""
from pathlib import Path
import argparse
import json
import sys
import time
import tempfile
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pc.multiobject.yolo import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--side', type=int, default=384)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--device', choices=['cpu', 'cuda', 'rocm'], default='cpu')
    a = p.parse_args()
    if a.out.exists() or min(a.epochs, a.batch, a.side) <= 0 or not 0 < a.learning_rate <= .01:
        p.error('Fresh output and positive dimensions required')
    if a.device != 'cpu' and (not torch.cuda.is_available() or bool(torch.version.hip) != (a.device == 'rocm')):
        p.error('Requested accelerator unavailable; fallback refused')
    from ultralytics import YOLO
    torch.set_num_threads(6)
    data = json.loads((a.dataset/'dataset.json').read_bytes())
    if not {'red_cube', 'yellow_cube', 'blue_cube'}.issubset(data['trained_classes']):
        p.error('All three colour classes need training examples')
    start = time.perf_counter()
    model = YOLO(str(a.weights))
    import yaml
    # Regenerate paths on this host; a copied Ultralytics YAML may name the old PC.
    with tempfile.TemporaryDirectory(prefix='multiobject_train_') as temporary:
        yaml_file = Path(temporary)/'data.yaml'
        yaml_file.write_text(yaml.safe_dump({'path': str(a.dataset.resolve()),
            'train': 'images/train', 'val': 'images/validation', 'test': 'images/test',
            'names': dict(enumerate(data['classes']))}), encoding='utf8')
        model.train(data=str(yaml_file), epochs=a.epochs, imgsz=a.side, batch=a.batch,
                    device='cpu' if a.device == 'cpu' else 0, workers=0, project=str(a.out.parent.resolve()),
                    name=a.out.name, exist_ok=False, pretrained=True, seed=20261025, deterministic=True,
                    amp=False, plots=False, freeze=10, hsv_h=0., hsv_s=.1, hsv_v=.4,
                    mosaic=.5, close_mosaic=5, mixup=0., fliplr=.5, flipud=.2,
                    degrees=30., scale=.3, translate=.15, patience=15, optimizer='AdamW', lr0=a.learning_rate)
    report = {'schema': 'gesture.ai_pc.multiobject_training.v1',
              'dataset_sha256': digest(a.dataset/'dataset.json'), 'pretrained_sha256': digest(a.weights),
              'classes': data['classes'], 'trained_classes': data['trained_classes'],
              'label_kinds': sorted(set(i.get('annotation_kind', 'UNSPECIFIED') for i in data['items'])),
              'epochs_requested': a.epochs, 'train_side': a.side, 'device': a.device,
              'learning_rate': a.learning_rate,
              'torch': torch.__version__, 'hip': torch.version.hip, 'elapsed_s': time.perf_counter()-start,
              'test_used_for_training': False, 'npu_validation': 'NOT_RUN', 'physical_execution': False}
    (a.out/'training_report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print('TRAINING_COMPLETED', a.out, flush=True)


if __name__ == '__main__':
    main()
