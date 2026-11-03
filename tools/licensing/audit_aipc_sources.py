"""Audit frozen AI PC model hashes and image-source license metadata, offline.

This validates provenance records; it does not grant rights or decide whether
trained weights are adaptations of copyrighted training images.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
FREEZE = 'artifacts/multiobject/eraser_top_repair_20261027/candidate_freeze_v2.json'
MANIFESTS = {'detector': 'handoff/detector_source_dataset.json',
             'surface': 'artifacts/multiobject/eraser_top_repair_20261027/dataset_v1/dataset.json'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def license_index(attributions, license_rows):
    licenses = {r['id']: r for r in license_rows}
    indexed = {}
    for row in attributions:
        if row['dataset'] != 'LVIS_V1':
            continue
        key = row['image_id']
        if row['coco_license_id'] not in licenses:
            raise ValueError('Unknown image license id: '+str(row['coco_license_id']))
        if key in indexed:
            previous = indexed[key]
            if any(previous[k] != row[k] for k in ('coco_license_id', 'image_sha256')):
                raise ValueError('Conflicting attribution for one image')
        indexed[key] = row
    return indexed, licenses


def get_source_ancestors(source):
    yield source
    parent = source.get('parent')
    if isinstance(parent, dict):
        yield from get_source_ancestors(parent)


def summarize(items, attribution, licenses):
    used = []
    originals = []
    for row in items:
        source = row['source']
        kind = source.get('kind')
        tops = sum(bool(o.get('top_polygon_raw')) for o in row['objects'])
        if kind == 'LVIS_V1_PUBLIC':
            origin = attribution[source['image_id']]
            if source['image_sha256'] != origin['image_sha256']:
                raise ValueError('LVIS original image hash disagrees with attribution')
            license_row = licenses[origin['coco_license_id']]
            used.append({
                'dataset': 'LVIS_COCO', 'image_id_or_path': str(source['image_id']),
                'source_image_sha256': source['image_sha256'], 'split': row['split'],
                'license_id': origin['coco_license_id'], 'license_name': license_row['name'],
                'license_url': license_row['url'], 'source_url': source['url'],
                'flickr_url': origin['flickr_url'], 'author_attribution_complete': False,
                'top_training_eligible': row['split'] == 'train' and tops > 0})
        elif kind == 'OFFICE_HOME_PUBLIC':
            used.append({
                'dataset': 'OFFICE_HOME', 'image_id_or_path': source['path'],
                'source_image_sha256': source['image_sha256'], 'split': row['split'],
                'license_id': '', 'license_name': 'OFFICIAL_NONCOMMERCIAL_RESEARCH_EDUCATION_FAIR_USE_NOTICE',
                'license_url': 'https://www.hemanthdv.org/officeHomeDataset.html',
                'source_url': source['url'], 'flickr_url': '',
                'author_attribution_complete': False,
                'top_training_eligible': row['split'] == 'train' and tops > 0})
        if kind in ('LVIS_V1_PUBLIC', 'OFFICE_HOME_PUBLIC'):
            originals.append(row)
    lvis = [r for r in used if r['dataset'] == 'LVIS_COCO']
    office = [r for r in used if r['dataset'] == 'OFFICE_HOME']
    counts = Counter(str(r['license_id']) for r in lvis)
    split_counts = {split: dict(sorted(Counter(str(r['license_id']) for r in lvis
                    if r['split'] == split).items())) for split in ('train', 'validation')}
    # The URL, not an assumed global ID convention, determines the condition.
    nc = [r for r in lvis if '/by-nc' in r['license_url']]
    nd = [r for r in lvis if '-nd/' in r['license_url']]
    inherited_office = [r for r in items if any(s.get('kind') == 'OFFICE_HOME_PUBLIC'
                         for s in get_source_ancestors(r['source']))]
    return {'items': len(items), 'splits': dict(Counter(r['split'] for r in items)),
            'lvis_original_images': len(lvis), 'lvis_license_id_counts': dict(sorted(counts.items())),
            'lvis_license_id_counts_by_split': split_counts,
            'lvis_noncommercial_image_count': len(nc), 'lvis_no_derivatives_image_count': len(nd),
            'condition_counts_overlap': True,
            'officehome_original_images': len(office),
            'officehome_original_splits': dict(Counter(r['split'] for r in office)),
            'officehome_original_or_direct_parent_derived_records': len(inherited_office),
            'officehome_original_train_with_top_polygon': sum(r['top_training_eligible'] for r in office),
            'lvis_original_train_with_top_polygon': sum(r['top_training_eligible'] for r in lvis),
            'surface_eligibility_note': 'Train + explicit top polygon is sampler eligibility, not a count of actual stochastic draws; no image-free layers inferred',
            'rights_clearance_from_metadata_alone': False}, used


def audit(root, mapping):
    root = Path(root).resolve()
    freeze = json.loads((root/FREEZE).read_bytes())
    checks = []
    for path, expected in freeze['files_sha256'].items():
        target = (root/path).resolve()
        if not target.is_relative_to(root) or not target.is_file() or sha(target) != expected:
            raise ValueError('Frozen file missing or changed: '+path)
        checks.append({'path': path, 'sha256': expected})
    original_attribution = root/'docs/phase3c/mode_b_six_objects_20261026/source_attribution.json'
    attribution = json.loads(original_attribution.read_bytes())['sources']
    mapping = Path(mapping)
    index, licenses = license_index(attribution, json.loads(mapping.read_bytes())['licenses'])
    office_by_url = {r['source_url']: r for r in attribution if r['dataset'] == 'OfficeHome'}
    report = {'schema': 'gesture.aipc.license_source_audit.v1', 'date': '2026-11-03',
              'scope': 'AI_PC_ONLY_NO_FPGA_OR_TEAMMATE_CLEARANCE',
              'metadata_integrity_passed': True, 'redistribution_clearance_complete': False,
              'license_granted_by_this_audit': False,
              'frozen_manifest_sha256': sha(root/FREEZE), 'frozen_files': checks,
              'source_attribution_sha256': sha(original_attribution),
              'official_license_mapping_sha256': sha(mapping), 'datasets': {},
              'model_weights_modified': False, 'training_started': False,
              'physical_execution': False, 'visibility_changed': False,
              'upstream_initializer_binary_identity_verified': False}
    source_rows = {}
    for role, name in MANIFESTS.items():
        data = json.loads((root/name).read_bytes())
        model_spec = json.loads((root/freeze[role+'_package']).read_bytes())
        declared_dataset = (model_spec.get('training_report') or {}).get('dataset_sha256') if role == 'detector' else model_spec['dataset_sha256']
        if sha(root/name) != declared_dataset:
            raise ValueError('Model training-dataset hash differs from audited manifest')
        for row in data['items']:
            source = row['source']
            if source.get('kind') == 'OFFICE_HOME_PUBLIC':
                original = office_by_url.get(source['url'])
                if not original or original['sha256'] != source['image_sha256']:
                    raise ValueError('Office-Home original hash/URL disagrees with attribution')
        summary, rows = summarize(data['items'], index, licenses)
        report['datasets'][role] = {'manifest': name, 'sha256': sha(root/name), **summary}
        for row in rows:
            key = (row['dataset'], row['source_image_sha256'])
            if key in source_rows and source_rows[key] != row:
                raise ValueError('Conflicting role source provenance')
            source_rows[key] = row
    return report, sorted(source_rows.values(), key=lambda r: (r['dataset'], r['image_id_or_path']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=REPOSITORY/'project')
    p.add_argument('--license-map', type=Path, default=REPOSITORY/'docs/aipc_license_20261103/lvis_official_image_license_mapping.json')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error('Fresh output directory required')
    report, rows = audit(args.project_root, args.license_map)
    args.out.mkdir(parents=True)
    (args.out/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (args.out/'used_public_image_sources.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'metadata_integrity_passed': True, 'public_originals': len(rows),
                      'redistribution_clearance_complete': False, 'datasets': report['datasets']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
