"""Negative integrity cases; tests never load weights or access devices."""
import hashlib,json
from pathlib import Path
import tempfile,unittest
from verify_runtime_assets import verify

class Tests(unittest.TestCase):
    def fixture(self,root):
        p=root/'weights.pt';p.write_bytes(b'test fixture, not actual weights')
        row=dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,role='TEST_ONLY')
        m=root/'manifest.json';m.write_text(json.dumps({'files':[row]}));return p,m,row
    def test_exact_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p,m,row=self.fixture(Path(d));self.assertTrue(verify(d,m)['integrity_passed'])
            p.write_bytes(b'x'*row['bytes']);self.assertEqual(verify(d,m)['results'][0]['status'],'HASH_MISMATCH')
    def test_missing_asset_is_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            p,m,row=self.fixture(Path(d));p.unlink();self.assertEqual(verify(d,m)['results'][0]['status'],'MISSING')
    def test_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p,m,row=self.fixture(Path(d));row['path']='../weights.pt';m.write_text(json.dumps({'files':[row]}))
            with self.assertRaises(ValueError):verify(d,m)
    def test_duplicate_manifest_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p,m,row=self.fixture(Path(d));m.write_text(json.dumps({'files':[row,row]}))
            with self.assertRaises(ValueError):verify(d,m)

if __name__=='__main__':unittest.main()
