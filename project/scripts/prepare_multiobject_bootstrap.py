"""Build weak top-box training data from TRAIN-only historical textures.

Not an independent detector dataset. Other five custom labels have NO positives.
"""
from pathlib import Path
import argparse
import json
import sys
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pc.multiobject.dataset import compile_dataset, write_image
from pc.multiobject.yolo import digest
from pc.neural_top_training import BootstrapTraining, load_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--samples', type=int, default=48)
    a = p.parse_args()
    if a.out.exists() or a.samples < 1:
        p.error('Fresh output directory and positive sample count required')
    source = ROOT/'datasets/neural_top_bootstrap_20261025_v1/dataset.json'
    original = load_dataset(source)
    training = BootstrapTraining(source, length=a.samples, side=384)
    staging = a.out/'annotations'
    staging.mkdir(parents=True)
    paths = []
    for index in range(a.samples+len(original['items'])):
        if index < a.samples:
            x, y = training[index]
            im = np.rint(x.numpy().transpose(1, 2, 0)[:, :, ::-1]*255).astype(np.uint8)
            mask = y.numpy().astype(np.uint8)
            objects = []
            for k in (1, 2, 3):
                n, labels, stats, _ = cv2.connectedComponentsWithStats((mask == k).astype(np.uint8))
                for j in range(1, n):
                    bx, by, bw, bh, area = map(int, stats[j])
                    if area >= 25:
                        objects.append({'class_name': ['red_cube', 'yellow_cube', 'blue_cube'][k-1],
                                        'bbox_xyxy_raw': [bx, by, bx+bw, by+bh]})
            session, split = 'synthetic_from_train_155202', 'train'
        else:
            from pc.multiobject.dataset import read_image
            item = original['items'][index-a.samples]
            im = read_image(source.parent/item['image'])
            session, split = item['session_id'], item['split']
            objects = []
            for obj in item['objects']:
                poly = np.asarray(obj['polygon_raw'])
                objects.append({'class_name': ['red_cube', 'yellow_cube', 'blue_cube'][obj['object_id']-1],
                                'bbox_xyxy_raw': [*poly.min(0).tolist(), *poly.max(0).tolist()],
                                'top_polygon_raw': obj['polygon_raw']})
        image_path = staging/f'item_{index:05d}.png'
        write_image(image_path, im)
        record = {'schema': 'gesture.ai_pc.multiobject_annotation.v1',
                  'session_id': session, 'split': split, 'image': image_path.name,
                  'image_sha256': digest(image_path), 'image_size_wh': [im.shape[1], im.shape[0]],
                  'objects': objects, 'empty_scene_confirmed': not objects,
                  'annotation_kind': 'HEURISTIC_TOP_BOX_BOOTSTRAP_NOT_FULL_OBJECT_GROUND_TRUTH'}
        path = staging/f'item_{index:05d}.json'
        path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf8')
        paths.append(path)
    result = compile_dataset(paths, ROOT/'config/ai_pc/multiobject_registry_v1.json', a.out/'dataset')
    print('BOOTSTRAP_READY', json.dumps(result['counts_by_split']), flush=True)


if __name__ == '__main__':
    main()
