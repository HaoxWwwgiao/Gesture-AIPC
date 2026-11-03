"""Append reviewed field bottle train sources; preserve all upstream splits."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pc.mode_b.data_audit import audit_dataset
from pc.mode_b.profile import CLASSES
from pc.multiobject.dataset import read_image, write_image
from pc.multiobject.public_data import sha256


def compile_dataset(base, annotations, out):
    base, annotations, out = map(Path, (base, annotations, out))
    if out.exists():
        raise ValueError('Fresh output required')
    audit_dataset(base / 'dataset.json')
    data = json.loads((base / 'dataset.json').read_bytes())
    field = json.loads(annotations.read_bytes())
    if field['split'] != 'train' or field['evaluation_images_included'] is not False:
        raise ValueError('Only explicit training annotations may be compiled')
    out.mkdir(parents=True)
    for kind in ('images', 'labels'):
        shutil.copytree(base / kind, out / kind)
    records = deepcopy(data['items'])
    original_count = len(records)
    rng = np.random.default_rng(20261027)
    for row in field['items']:
        if row['split'] != 'train' or not row['session_id']:
            raise ValueError('Training source family required')
        image_path = (ROOT / row['image']).resolve()
        if not image_path.is_relative_to(ROOT) or sha256(image_path) != row['image_sha256']:
            raise ValueError('Field original missing or changed')
        image = read_image(image_path)
        if hashlib.sha256(image.tobytes()).hexdigest() != row['pixel_sha256']:
            raise ValueError('Field pixels changed')
        objects = deepcopy(row['objects'])
        for obj in objects:
            if obj['class_name'] != 'small_canister':
                raise ValueError('This adaptation only accepts reviewed bottle sources')
        # A full frame plus eight context crops from the same train family.
        # Derivatives are never counted as independent captures or validation.
        variants = [(image, objects, None)]
        h, w = image.shape[:2]
        union = np.asarray([o['bbox_xyxy_raw'] for o in objects])
        lo, hi = union[:, :2].min(0), union[:, 2:].max(0)
        centre = (lo + hi) / 2
        for side in (360, 430, 510, 600, 700, 820, 950, 1050):
            side = min(side, h, w)
            origin = np.clip(np.rint(centre - side/2 + rng.uniform(-.16, .16, 2)*side),
                             0, [w-side, h-side]).astype(int)
            if np.any(lo-origin < 10) or np.any(hi-origin > side-10):
                raise ValueError('Context crop would truncate annotated bottle')
            shifted = deepcopy(objects)
            for obj in shifted:
                obj['bbox_xyxy_raw'] = (np.asarray(obj['bbox_xyxy_raw'])-np.tile(origin, 2)).tolist()
                obj['top_polygon_raw'] = (np.asarray(obj['top_polygon_raw'])-origin).tolist()
            x, y = origin
            variants.append((image[y:y+side, x:x+side].copy(), shifted,
                             [int(x), int(y), side, side]))
        for im, objs, crop in variants:
            name = f'bottle_{len(records)-original_count:04d}'
            path = out / 'images/train' / (name+'.png')
            write_image(path, im)
            ih, iw = im.shape[:2]
            labels = []
            for obj in objs:
                x, y, x2, y2 = obj['bbox_xyxy_raw']
                labels.append(f'{CLASSES.index(obj["class_name"])} {(x+x2)/2/iw:.9f} '
                              f'{(y+y2)/2/ih:.9f} {(x2-x)/iw:.9f} {(y2-y)/ih:.9f}')
            (out / 'labels/train' / (name+'.txt')).write_text('\n'.join(labels)+'\n', encoding='utf8')
            records.append({'image':path.relative_to(out).as_posix(), 'image_sha256':sha256(path),
                'pixel_sha256':hashlib.sha256(im.tobytes()).hexdigest(), 'image_size_wh':[iw, ih],
                'objects':objs, 'split':'train', 'session_id':row['session_id'],
                'source':{'kind':'PROJECT_MEDICINE_BOTTLE_FIELD' if crop is None else 'TRAIN_CONTEXT_CROP',
                          'parent_image':row['image'], 'parent_sha256':row['image_sha256'],
                          'crop_xywh':crop, 'independent_capture':crop is None},
                'annotation_kind':'ASSISTANT_REVIEWED_BOTTLE_BOX_AND_APPROXIMATE_CAP_CONTOUR'})
    data.update(items=records, counts_by_split={s:dict(Counter(o['class_name'] for r in records
        if r['split']==s for o in r['objects'])) for s in ('train','validation')},
        parent_manifest_sha256=sha256(base/'dataset.json'), field_annotations_sha256=sha256(annotations),
        field_original_images=len(field['items']), field_train_derivatives=len(records)-original_count-len(field['items']),
        source_split_policy='ALL_UPSTREAM_SPLITS_PRESERVED_FIELD_SOURCES_AND_DERIVATIVES_TRAIN_ONLY',
        metric_scope='PUBLIC_DEVELOPMENT_PLUS_BOTTLE_TRAIN_SOURCES_NOT_FIELD_ACCEPTANCE')
    (out/'dataset.json').write_text(json.dumps(data, indent=2)+'\n', encoding='utf8')
    (out/'data.yaml').write_text(yaml.safe_dump({'path':str(out.resolve()), 'train':'images/train',
        'val':'images/validation', 'names':dict(enumerate(CLASSES))}), encoding='utf8')
    audit = audit_dataset(out/'dataset.json')
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf8')
    return audit


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(compile_dataset(a.base, a.annotations, a.out)))
