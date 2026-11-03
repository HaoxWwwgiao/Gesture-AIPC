"""Train a binary cube-top refiner; only TRAIN sessions supply gradients."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TORCH_HOME', str(ROOT/'artifacts/multiobject/source/torch_cache'))
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pc.multiobject.surface import SurfaceNet
from pc.multiobject.yolo import digest
from pc.neural_top_training import BootstrapTraining


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=250)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--manual-dataset', type=Path, help='Compiled independent boxes+four-corner cube tops')
    p.add_argument('--weights', type=Path, help='Warm start an existing SurfaceNet checkpoint')
    p.add_argument('--learning-rate', type=float, default=.002)
    a = p.parse_args()
    if a.out.exists() or min(a.steps, a.batch) <= 0 or not 0 < a.learning_rate <= .01:
        p.error('Fresh output and positive training dimensions required')
    torch.set_num_threads(6)
    torch.manual_seed(20261025)
    data = a.manual_dataset or ROOT/'datasets/neural_top_bootstrap_20261025_v1/dataset.json'
    side = 256 if a.manual_dataset else 160
    if a.manual_dataset:
        from pc.multiobject.surface_training import ManualSurfaceTraining
        from pc.multiobject.registry import Registry
        ds = ManualSurfaceTraining(data, Registry.load(ROOT/'config/ai_pc/multiobject_registry_v1.json'),
                                   a.steps*a.batch, side=side)
    else:
        ds = BootstrapTraining(data, length=a.steps*a.batch, side=side)
    model = SurfaceNet(pretrained=True)
    if a.weights:
        model.load_state_dict(torch.load(a.weights, map_location='cpu', weights_only=True), strict=True)
    for par in model.encoder.parameters():
        par.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.learning_rate)
    a.out.mkdir(parents=True)
    start = time.perf_counter()
    with (a.out/'training.jsonl').open('w', encoding='utf8') as log:
        for step, (x, y) in enumerate(DataLoader(ds, batch_size=a.batch, num_workers=0), 1):
            model.train(); model.encoder.eval()
            target = (y > 0).float()[:, None]
            logits = model(x)
            pred = logits.sigmoid()
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(2.))
            loss = loss+.5*(1-(2*(pred*target).sum()+1)/(pred.sum()+target.sum()+1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            row = {'step': step, 'loss': float(loss.detach()), 'elapsed_s': time.perf_counter()-start}
            log.write(json.dumps(row)+'\n')
            if step == 1 or step % 25 == 0:
                print(json.dumps(row), flush=True); log.flush()
    model.eval()
    torch.save(model.state_dict(), a.out/'weights.pt')
    spec = {'schema': 'gesture.ai_pc.surface_package.v1', 'architecture': SurfaceNet.architecture,
            'input_encoding': 'RGB_FLOAT_0_1_INTERNAL_IMAGENET_NORMALIZATION', 'input_side': 256,
            'checkpoint': {'file': 'weights.pt', 'sha256': digest(a.out/'weights.pt')},
            'encoder_source': 'torchvision MobileNet_V3_Small_Weights.IMAGENET1K_V1',
            'encoder_frozen': True, 'steps': a.steps, 'batch': a.batch, 'train_side': side,
            'learning_rate': a.learning_rate,
            'warm_start_sha256': digest(a.weights) if a.weights else None,
            'dataset_sha256': digest(data), 'training_labels': 'ANNOTATED_POLYGONS_TRAIN_ONLY_REQUIRES_REVIEW' if a.manual_dataset else 'HEURISTIC_BOOTSTRAP_NOT_INDEPENDENT',
            'annotation_kinds': sorted({i.get('annotation_kind', 'UNSPECIFIED') for i in ds.data['items']}) if a.manual_dataset else ['HEURISTIC_BOOTSTRAP'],
            'training_device': 'CPU', 'rocm_validation': 'NOT_RUN',
            'elapsed_s': time.perf_counter()-start, 'physical_execution': False}
    (a.out/'model.json').write_text(json.dumps(spec, indent=2)+'\n', encoding='utf8')
    print('SURFACE_TRAINED', a.out, flush=True)


if __name__ == '__main__':
    main()
