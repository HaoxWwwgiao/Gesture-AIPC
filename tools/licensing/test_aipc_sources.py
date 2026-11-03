"""Provenance failures that must stop an AI PC source-license audit."""
import unittest
from audit_aipc_sources import license_index, summarize


class SourceAuditTests(unittest.TestCase):
    def test_same_photo_annotations_must_not_disagree_on_license(self):
        row = {'dataset': 'LVIS_V1', 'image_id': 1, 'coco_license_id': 4, 'image_sha256': 'a'}
        licenses = [{'id': 4}, {'id': 6}]
        index, _ = license_index([row, dict(row, annotation_id=2)], licenses)
        self.assertEqual(len(index), 1)
        with self.assertRaises(ValueError):
            license_index([row, dict(row, coco_license_id=6)], licenses)

    def test_unknown_license_id_is_never_silently_permissive(self):
        with self.assertRaises(ValueError):
            license_index([{'dataset': 'LVIS_V1', 'image_id': 1, 'coco_license_id': 99}], [])

    def test_edited_image_source_hash_stops_audit(self):
        item = {'source': {'kind': 'LVIS_V1_PUBLIC', 'image_id': 1, 'image_sha256': 'wrong'},
                'objects': [], 'split': 'train'}
        with self.assertRaises(ValueError):
            summarize([item], {1: {'image_sha256': 'original'}}, {})

    def test_derivative_retains_parent_notice_and_top_eligibility_is_separate(self):
        source = {'kind': 'OFFICE_HOME_PUBLIC', 'path': 'eraser.jpg', 'image_sha256': 'a', 'url': 'source'}
        items = [{'source': source, 'split': 'train', 'objects': [{'bbox': [0, 0, 1, 1]}]},
                 {'source': {'kind': 'SYNTHETIC', 'parent': source}, 'split': 'train',
                  'objects': [{'top_polygon_raw': [[0, 0], [1, 0], [1, 1], [0, 1]]}]}]
        report, images = summarize(items, {}, {})
        self.assertEqual(report['officehome_original_or_direct_parent_derived_records'], 2)
        self.assertEqual(report['officehome_original_train_with_top_polygon'], 0)
        self.assertEqual(len(images), 1)
        self.assertFalse(report['rights_clearance_from_metadata_alone'])


if __name__ == '__main__':
    unittest.main()
