"""Compile reviewed old camera images; never import frozen holdout families into train."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pc.multiobject.history import build_history

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--baseline', type=Path, default=ROOT/'artifacts/multiobject/bootstrap_v1/dataset/dataset.json')
    p.add_argument('--registry', type=Path, default=ROOT/'config/ai_pc/multiobject_registry_v1.json')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build_history(a.review, a.baseline, a.registry, ROOT, a.out)), flush=True)
