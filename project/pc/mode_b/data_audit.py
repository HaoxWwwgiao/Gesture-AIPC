"""Audit six-class source splits and actual image content before training/export."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from pc.mode_b.profile import CLASSES
from pc.multiobject.dataset import read_image
from pc.multiobject.public_data import sha256


def audit_dataset(manifest):
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_bytes())
    if (data.get('schema') != 'gesture.ai_pc.mode_b_dataset.v1'
            or data.get('classes') != CLASSES
            or data.get('test_used_for_training') is not False):
        raise ValueError('Explicit six-class development manifest required')
    groups, files, pixels = {}, {}, {}
    counts = defaultdict(Counter)
    image_counts = Counter()
    source_counts = Counter()
    top_counts = defaultdict(Counter)
    for row in data['items']:
        split = row['split']
        if split not in ('train', 'validation'):
            raise ValueError('This public development dataset must not consume test data')
        path = (manifest.parent / row['image']).resolve()
        if not path.is_relative_to(manifest.parent) or not path.is_file():
            raise ValueError('Dataset image path escapes or is missing')
        file_hash = sha256(path)
        image = read_image(path)
        pixel_hash = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        if file_hash != row['image_sha256'] or pixel_hash != row['pixel_sha256']:
            raise ValueError('Dataset image content changed')
        h, w = image.shape[:2]
        if row['image_size_wh'] != [w, h]:
            raise ValueError('Dataset image dimensions changed')
        for mapping, key in ((groups, row['session_id']), (files, file_hash), (pixels, pixel_hash)):
            if key in mapping and mapping[key] != split:
                raise ValueError('Source family or identical image crosses split')
            mapping[key] = split
        source_kind = row['source']['kind']
        if source_kind == 'SYNTHETIC_COPY_PASTE_FROM_TRAIN_ONLY' and split != 'train':
            raise ValueError('Derived augmentation must remain on training side')
        image_counts[split] += 1
        source_counts[source_kind] += 1
        for obj in row['objects']:
            if obj['class_name'] not in CLASSES:
                raise ValueError('Unexpected class in mode B dataset')
            box = np.asarray(obj['bbox_xyxy_raw'], dtype=float)
            if (box.shape != (4,) or not np.isfinite(box).all()
                    or np.any(box[:2] < 0) or np.any(box[2:] > [w, h])
                    or np.any(box[2:] <= box[:2])):
                raise ValueError('Invalid box annotation')
            counts[split][obj['class_name']] += 1
            if obj.get('top_polygon_raw'):
                polygon = np.asarray(obj['top_polygon_raw'], dtype=float)
                if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                        or not np.isfinite(polygon).all() or np.any(polygon < 0)
                        or np.any(polygon > [w, h])):
                    raise ValueError('Invalid top polygon')
                top_counts[split][obj['class_name']] += 1
    if {s: dict(c) for s, c in counts.items()} != data['counts_by_split']:
        raise ValueError('Manifest counts do not match annotations')
    if set(counts['train']) != set(CLASSES):
        raise ValueError('Every selected class needs actual training examples')
    return {
        'status': 'CONTENT_AND_SPLIT_AUDIT_PASS',
        'manifest_sha256': sha256(manifest),
        'image_counts': dict(image_counts),
        'counts_by_split': {s: dict(c) for s, c in counts.items()},
        'source_image_counts': dict(source_counts),
        'surface_counts_including_derivatives': {s: dict(c) for s, c in top_counts.items()},
        'unique_source_groups': len(groups),
        'test_consumed': False,
        'perceptual_near_duplicates_excluded': False,
        'public_annotation_completeness_independently_verified': False,
        'physical_validation': 'NOT_RUN',
    }
