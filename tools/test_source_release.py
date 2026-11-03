"""Negative packaging checks, using temporary source-only fixtures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import hashlib
import verify_source_release
from verify_source_release import inspect_file,verify

class Tests(unittest.TestCase):
    def test_binary_and_embedded_assets_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name,raw in [('model.onnx',b'weights'),('fake.py',b'x=1\0'),('payload.html',('data:image/png;base64,'+'A'*200).encode())]:
                path=root/name;path.write_bytes(raw)
                with self.assertRaises(ValueError):inspect_file(path,name)
    def test_valid_source_and_changed_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'sample.py';p.write_text('x=1\n')
            row=inspect_file(p,'sample.py');(root/'source_manifest.json').write_text(json.dumps({'files':[row]}))
            self.assertTrue(verify(root)['source_subset_passed'])
            p.write_text('x=2\n')
            with self.assertRaises(ValueError):verify(root)
    def test_manifest_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'source_manifest.json').write_text(json.dumps({'files':[{'path':'../outside.py'}]}))
            with self.assertRaises(ValueError):verify(root)
    def test_disguised_numeric_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'model.json';p.write_text(json.dumps({'state_dict':{'layer.weight':[0.1]*256}}))
            with self.assertRaises(ValueError):inspect_file(p,'model.json')
    def test_only_exact_authorized_model_bytes_are_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'fixture.pt';payload=b'fixture';p.write_bytes(payload)
            name='project/model_package/test/fixture.pt'
            with patch.dict(verify_source_release.MODELS,{name:(len(payload),hashlib.sha256(payload).hexdigest())}):
                self.assertEqual(inspect_file(p,name)['role'],'AUTHORIZED_FROZEN_MODEL')
                p.write_bytes(b'changed')
                with self.assertRaises(ValueError):inspect_file(p,name)

if __name__=='__main__':unittest.main()
