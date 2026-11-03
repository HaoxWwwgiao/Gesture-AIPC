"""Verify an independent inference inventory; no downloads, models or devices opened."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath

def verify(root,manifest):
    root=Path(root).resolve();data=json.loads(Path(manifest).read_bytes());rows=[]
    names=set()
    for entry in data['files']:
        name=entry['path'];part=PurePosixPath(name)
        if part.is_absolute() or '..' in part.parts or ':' in name or '\\' in name or name in names:
            raise ValueError('Unsafe or duplicate runtime path: '+name)
        names.add(name);path=(root/name).resolve()
        if not path.is_relative_to(root):raise ValueError('Escaping runtime path: '+name)
        if not path.is_file():status='MISSING'
        elif path.stat().st_size!=entry['bytes']:status='SIZE_MISMATCH'
        else:
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            status='MATCH' if digest==entry['sha256'] else 'HASH_MISMATCH'
        rows.append(dict(path=name,status=status,role=entry['role']))
    return dict(schema='gesture.aipc.runtime_asset_check.v1',integrity_passed=all(r['status']=='MATCH' for r in rows),
                files=len(rows),results=rows,license_clearance_granted=False,hardware_access=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('MINIMUM_VISION_RUNTIME.json'));a=p.parse_args()
    result=verify(a.root,a.manifest);print(json.dumps(result,indent=2));return 0 if result['integrity_passed'] else 2

if __name__=='__main__':raise SystemExit(main())
