"""Offline source inventory check. No model loading, downloads or device access."""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ALLOWED={'.py','.ps1','.tcl','.cs','.js','.cjs','.mjs','.css','.html','.svg','.md','.json','.txt','.csv','.yaml','.yml'}
SPECIAL={'LICENSE','NOTICE','.gitignore','.gitattributes'}
SKIP={'.git','.venv','__pycache__','_local'}
SECRETS=re.compile(rb'github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|AKIA[0-9A-Z]{16}')

def inspect_file(path, name):
    if path.is_symlink():raise ValueError('Symbolic link: '+name)
    if path.suffix.lower() not in ALLOWED and name not in SPECIAL:raise ValueError('Non-source file: '+name)
    raw=path.read_bytes()
    if len(raw)>2_000_000:raise ValueError('Oversized source: '+name)
    if b'\0' in raw:raise ValueError('Binary payload: '+name)
    text=raw.decode('utf-8-sig')
    if SECRETS.search(raw):raise ValueError('Possible credential: '+name)
    if re.search(r'data:(?:image|application/octet-stream)[^,]*;base64,[A-Za-z0-9+/]{128}',text):raise ValueError('Embedded binary: '+name)
    if path.suffix=='.py':compile(raw,name,'exec')
    if path.suffix=='.json':
        data=json.loads(raw)
        def inspect_json(value,key=''):
            if isinstance(value,dict):
                for child,item in value.items():inspect_json(item,str(child))
            elif isinstance(value,list):
                numeric=all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in value)
                if numeric and (len(value)>4096 or (len(value)>64 and re.search(r'weight|bias|tensor|initializer',key,re.I))):
                    raise ValueError('Possible numeric tensor payload: '+name)
                for item in value:
                    if isinstance(item,(dict,list)):inspect_json(item,key)
        inspect_json(data)
    return {'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def verify(root):
    root=Path(root).resolve()
    inspect_file(root/'source_manifest.json','source_manifest.json')
    manifest=json.loads((root/'source_manifest.json').read_bytes())
    actual={}
    for p in root.rglob('*'):
        rel=p.relative_to(root)
        if any(part in SKIP for part in rel.parts):continue
        if p.is_symlink():raise ValueError('Symbolic link: '+rel.as_posix())
        if not p.is_file() or rel.as_posix()=='source_manifest.json':continue
        row=inspect_file(p,rel.as_posix());actual[row['path']]=row
    expected={}
    for row in manifest['files']:
        name=row['path'];pure=PurePosixPath(name)
        if pure.is_absolute() or '..' in pure.parts or '\\' in name or ':' in name:raise ValueError('Unsafe manifest path')
        if name in expected:raise ValueError('Duplicate inventory path')
        expected[name]=row
    if set(actual)!=set(expected):raise ValueError('Inventory missing/extra: '+str(sorted(set(actual)^set(expected))))
    for name,row in actual.items():
        if row['sha256']!=expected[name]['sha256'] or row['bytes']!=expected[name]['bytes']:raise ValueError('Changed source: '+name)
    return dict(source_subset_passed=True,files=len(actual),python_files=sum(n.endswith('.py') for n in actual),
                runtime_ready=False,model_weights_included=False,training_images_included=False,
                hardware_access=False,npu_acceptance='NOT_RUN',full_runtime_assets='EXTERNAL_REQUIRED')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=ap.parse_args()
    print(json.dumps(verify(args.root),indent=2))

if __name__=='__main__':main()
