"""Compile reviewed six-object field sources, keeping context objects labelled."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pc.mode_b.data_audit import audit_dataset
from pc.mode_b.profile import CLASSES
from pc.multiobject.dataset import read_image, write_image
from pc.multiobject.public_data import sha256


def crop_objects(objects, crop):
    """Keep every intersecting detector label; train tops only when untruncated."""
    x, y, w, h = crop
    output = []
    for original in objects:
        box = np.asarray(original['bbox_xyxy_raw'], dtype=float)
        lo, hi = np.maximum(box[:2], [x, y]), np.minimum(box[2:], [x+w, y+h])
        if np.any(hi <= lo):
            continue
        obj = deepcopy(original)
        obj['bbox_xyxy_raw'] = np.r_[lo-[x, y], hi-[x, y]].tolist()
        truncated = not (np.all(box[:2] >= [x, y]) and np.all(box[2:] <= [x+w, y+h]))
        obj['truncated_by_context_crop'] = truncated
        if original.get('top_polygon_raw'):
            polygon = np.asarray(original['top_polygon_raw'], dtype=float)-[x, y]
            if truncated or np.any(polygon < 0) or np.any(polygon > [w-1, h-1]):
                obj.pop('top_polygon_raw', None)
            else:
                obj['top_polygon_raw'] = polygon.tolist()
        output.append(obj)
    return output


def compile_dataset(base, annotations, out):
    base, annotations, out = map(Path, (base, annotations, out))
    if out.exists():
        raise ValueError('Fresh output required')
    audit_dataset(base/'dataset.json')
    data = json.loads((base/'dataset.json').read_bytes())
    field = json.loads(annotations.read_bytes())
    if (field.get('split') != 'train' or field.get('evaluation_images_included') is not False
            or any(r.get('split') != 'train' for r in field['items'])):
        raise ValueError('Explicit train-only field manifest required')
    records = deepcopy(data['items'])
    out.mkdir(parents=True)
    for kind in ('images', 'labels'):
        shutil.copytree(base/kind, out/kind)
    seen = {r['pixel_sha256'] for r in records}
    added_originals, added_crops = 0, 0
    rng = np.random.default_rng(20261027)
    for row in field['items']:
        path = (ROOT/row['image']).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != row['image_sha256']:
            raise ValueError('Field original content changed')
        image = read_image(path)
        if hashlib.sha256(image.tobytes()).hexdigest() != row['pixel_sha256']:
            raise ValueError('Field pixel content changed')
        h, w = image.shape[:2]
        variants = [(image, deepcopy(row['objects']), None)]
        for target in row['objects']:
            box = np.asarray(target['bbox_xyxy_raw'])
            for requested in (640, 960, 1200):
                side = min(requested, h, w)
                if np.max(box[2:]-box[:2])+30 > side:
                    continue
                centre = (box[:2]+box[2:])/2
                origin = np.clip(np.rint(centre-side/2+rng.uniform(-.08,.08,2)*side),
                                 0, [w-side, h-side]).astype(int)
                if np.any(box[:2]-origin < 5) or np.any(box[2:]-origin > side-5):
                    continue
                x, y = map(int, origin)
                crop = [x, y, side, side]
                variants.append((image[y:y+side, x:x+side].copy(), crop_objects(row['objects'], crop), crop))
        for variant_index, (im, objects, crop) in enumerate(variants):
            pixel = hashlib.sha256(im.tobytes()).hexdigest()
            if pixel in seen:
                if crop is None:
                    raise ValueError('Field original duplicates existing dataset pixels')
                continue
            seen.add(pixel)
            # Successive compilations inherit prior field files. Bind new names
            # to their original instead of restarting a shared numeric counter.
            name = f'sixfield_{row["image_sha256"][:16]}_{variant_index:04d}'
            target_path = out/'images/train'/(name+'.png')
            label_path = out/'labels/train'/(name+'.txt')
            if target_path.exists() or label_path.exists():
                raise FileExistsError('Field output name already exists; inherited data must not be overwritten')
            write_image(target_path, im)
            ih, iw = im.shape[:2]
            labels = []
            for obj in objects:
                x, y, x2, y2 = obj['bbox_xyxy_raw']
                labels.append(f'{CLASSES.index(obj["class_name"])} {(x+x2)/2/iw:.9f} '
                              f'{(y+y2)/2/ih:.9f} {(x2-x)/iw:.9f} {(y2-y)/ih:.9f}')
            with label_path.open('x', encoding='utf8') as stream:
                stream.write('\n'.join(labels)+'\n')
            records.append({'image':target_path.relative_to(out).as_posix(),
                'image_sha256':sha256(target_path), 'pixel_sha256':pixel, 'image_size_wh':[iw, ih],
                'objects':objects, 'split':'train', 'session_id':row['session_id'],
                'source':{'kind':'PROJECT_SIX_OBJECT_FIELD' if crop is None else 'TRAIN_SIX_OBJECT_CONTEXT_CROP',
                          'parent_image':row['image'], 'parent_sha256':row['image_sha256'],
                          'crop_xywh':crop, 'independent_capture':crop is None},
                'annotation_kind':'ASSISTANT_VISUAL_BOXES_AND_VISIBLE_TOP_CONTOURS'})
            added_originals += int(crop is None)
            added_crops += int(crop is not None)
    data.update(items=records, counts_by_split={s:dict(Counter(o['class_name'] for r in records
        if r['split']==s for o in r['objects'])) for s in ('train','validation')},
        parent_manifest_sha256=sha256(base/'dataset.json'), field_annotations_sha256=sha256(annotations),
        added_field_original_images=added_originals, added_field_train_derivatives=added_crops,
        source_split_policy='UPSTREAM_SPLITS_UNCHANGED_FIELD_AND_CONTEXT_TRAIN_ONLY',
        metric_scope='SIX_CLASS_FIELD_ADAPTATION_DEVELOPMENT_NOT_ACCEPTANCE')
    (out/'dataset.json').write_text(json.dumps(data, indent=2)+'\n', encoding='utf8')
    (out/'data.yaml').write_text(yaml.safe_dump({'path':str(out.resolve()), 'train':'images/train',
        'val':'images/validation', 'names':dict(enumerate(CLASSES))}), encoding='utf8')
    audit = audit_dataset(out/'dataset.json')
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf8')
    return audit


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'annotations', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(compile_dataset(a.base, a.annotations, a.out)))
