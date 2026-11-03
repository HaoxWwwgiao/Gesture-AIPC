"""Index official LVIS metadata and fetch only proposed target images for review."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import requests
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pc.multiobject.public_data import index_lvis, select_instances, sha256
from pc.multiobject.dataset import read_image, write_image


def fetch(proposal, source):
    im = proposal['image']
    original = im['coco_url']
    # AWS path-style HTTPS has a valid certificate, unlike the dotted bucket's
    # images.cocodataset.org HTTPS endpoint on some networks.
    suffix = original.split('images.cocodataset.org/', 1)[1]
    url = 'https://s3.amazonaws.com/images.cocodataset.org/'+suffix
    out = source/(str(im['id'])+'.jpg')
    if not out.exists():
        r = requests.get(url, timeout=(8, 20))
        r.raise_for_status()
        if len(r.content) > 15*1024*1024:
            raise ValueError('Unexpected image size')
        decoded = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape[:2] != (im['height'], im['width']):
            raise ValueError('Upstream image size/encoding mismatch')
        out.write_bytes(r.content)
    return {**proposal, 'local_image': out.relative_to(ROOT).as_posix(),
            'image_sha256': sha256(out), 'download_url': url,
            'source_url': original, 'top_label': None,
            'label_scope': 'UPSTREAM_FULL_INSTANCE_NOT_TOP_FACE'}


def sheets(records, output, split):
    for name in sorted({r['class_name'] for r in records}):
        rows = [r for r in records if r['class_name'] == name]
        for page in range((len(rows)+23)//24):
            sheet = np.full((4*230, 6*220, 3), 238, np.uint8)
            for j, row in enumerate(rows[page*24:(page+1)*24]):
                image = read_image(ROOT/row['local_image'])
                x,y,w,h = row['annotation']['bbox']; pad = .15*max(w,h)
                x,y,x2,y2 = int(max(0,x-pad)),int(max(0,y-pad)),int(min(image.shape[1],x+w+pad)),int(min(image.shape[0],y+h+pad))
                crop = image[y:y2,x:x2]
                scale = min(210/crop.shape[1],195/crop.shape[0])
                thumb = cv2.resize(crop,(max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale))))
                sy,sx = (j//6)*230,(j%6)*220
                sheet[sy:sy+thumb.shape[0],sx:sx+thumb.shape[1]]=thumb
                cv2.putText(sheet,str(row['annotation']['id']),(sx+2,sy+216),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1)
            write_image(output/f'{split}_{name}_{page}.png',sheet)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'artifacts/multiobject/mode_b_public_20261026')
    p.add_argument('--split',choices=['train','val'],required=True)
    p.add_argument('--limit-per-class',type=int,default=48)
    a=p.parse_args(); source=a.root/'source'; source.mkdir(parents=True,exist_ok=True)
    index_path=a.root/f'lvis_{a.split}_index.json'
    if index_path.exists(): index=json.loads(index_path.read_bytes())
    else:
        index=index_lvis(source/f'lvis_v1_{a.split}.json.zip')
        index_path.write_text(json.dumps(index,indent=2)+'\n',encoding='utf8')
    proposals=select_instances(index,a.limit_per_class)
    print(json.dumps({'stage':'INDEXED','counts':index['counts'],'proposals':len(proposals)}),flush=True)
    errors=[]; records=[]
    # Deduplicate concurrent requests for the same image.
    unique={r['image']['id']:r for r in proposals}
    def download_one(row):
        try:return fetch(row,source)
        except Exception as e:return {'error':str(e),'image_id':row['image']['id']}
    fetched={}
    with ThreadPoolExecutor(6) as pool:
        for row in pool.map(download_one,unique.values()):
            if 'error' in row:errors.append(row)
            else:fetched[row['image']['id']]=row
    for row in proposals:
        if row['image']['id'] in fetched:
            downloaded=fetched[row['image']['id']]
            records.append({**downloaded,**row})
    result={'schema':'gesture.ai_pc.public_review_proposals.v1','upstream_split':a.split,
            'records':records,'download_errors':errors,'visual_review_complete':False,
            'upstream_instance_masks_used_as_top_labels':False}
    (a.root/f'{a.split}_proposals.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    output=a.root/'contact_sheets';output.mkdir(exist_ok=True)
    sheets(records,output,a.split)
    print(json.dumps({'stage':'REVIEW_READY','records':len(records),'errors':errors}),flush=True)


if __name__=='__main__':main()
