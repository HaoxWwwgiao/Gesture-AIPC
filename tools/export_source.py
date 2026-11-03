"""Export checked source and authorized frozen models; no Git history or datasets."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from verify_source_release import verify

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();root=args.root.resolve();out=args.out.resolve()
    result=verify(root)
    if out.exists():ap.error('Output must not exist')
    names=[r['path'] for r in json.loads((root/'source_manifest.json').read_bytes())['files']]+['source_manifest.json']
    out.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(out,'x',compression=ZIP_DEFLATED) as archive:
        for name in sorted(names):archive.write(root/name,name)
    with ZipFile(out) as archive:
        if archive.testzip() is not None:raise ValueError('ZIP CRC mismatch')
        if set(archive.namelist())!=set(names):raise ValueError('ZIP inventory mismatch')
        for name in names:
            if archive.read(name)!=(root/name).read_bytes():raise ValueError('ZIP byte mismatch: '+name)
    result.update(zip_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),zip_bytes=out.stat().st_size,zip_members=len(names),git_history_included=False)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
