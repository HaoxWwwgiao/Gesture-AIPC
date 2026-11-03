"""Add reviewed background crops from complete project TRAIN annotations only."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pc.mode_b.data_audit import audit_dataset
from pc.multiobject.dataset import read_image,write_image
from pc.multiobject.public_data import sha256


def background_is_clear(crop,objects,margin=4):
    x,y,w,h=crop
    return all(min(x+w,b[2]+margin)<=max(x,b[0]-margin)
               or min(y+h,b[3]+margin)<=max(y,b[1]-margin)
               for b in (o['bbox_xyxy_raw'] for o in objects))


def compile_background(base,out,per_source=8):
    base,out=Path(base),Path(out)
    if out.exists():raise ValueError('Fresh output required')
    audit_dataset(base/'dataset.json')
    data=json.loads((base/'dataset.json').read_bytes())
    records=deepcopy(data['items'])
    out.mkdir(parents=True)
    for kind in ('images','labels'):shutil.copytree(base/kind,out/kind)
    rng=np.random.default_rng(20261027)
    seen={r['pixel_sha256'] for r in records}
    added=[]
    for row in data['items']:
        if (row['split']!='train' or row['source']['kind'] not in
                ('PROJECT_SIX_OBJECT_FIELD','PROJECT_MEDICINE_BOTTLE_FIELD')):
            continue
        image=read_image(base/row['image']);h,w=image.shape[:2]
        choices=[]
        for side in (128,256,384,512):
            if side>min(h,w):continue
            for x in (0,w-side):
                for y in (0,(h-side)//2,h-side):choices.append([x,y,side,side])
        for _ in range(100):
            cw=int(rng.integers(128,min(w,700)+1));ch=int(rng.integers(128,min(h,700)+1))
            choices.append([int(rng.integers(0,w-cw+1)),int(rng.integers(0,h-ch+1)),cw,ch])
        count=0
        for crop in choices:
            if not background_is_clear(crop,row['objects']):continue
            x,y,cw,ch=crop;im=image[y:y+ch,x:x+cw].copy()
            if float(np.std(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)))<8:continue
            pixel=hashlib.sha256(im.tobytes()).hexdigest()
            if pixel in seen:continue
            seen.add(pixel)
            name='background_'+pixel[:20]
            dest=out/'images/train'/(name+'.png');label=out/'labels/train'/(name+'.txt')
            if dest.exists() or label.exists():raise FileExistsError('Do not overwrite inherited data')
            write_image(dest,im);label.open('x',encoding='utf8').close()
            item={'image':dest.relative_to(out).as_posix(),'image_sha256':sha256(dest),
                  'pixel_sha256':pixel,'image_size_wh':[cw,ch],'objects':[],'split':'train',
                  'session_id':row['session_id'],
                  'source':{'kind':'TRAIN_BACKGROUND_FROM_COMPLETE_PROJECT_BOXES',
                            'parent_image':row['image'],'parent_sha256':row['image_sha256'],
                            'crop_xywh':crop,'independent_capture':False},
                  'annotation_kind':'BOX_EXCLUSION_BACKGROUND_REQUIRES_CONTACT_SHEET_REVIEW'}
            records.append(item);added.append(item);count+=1
            if count>=per_source:break
    if not added:raise ValueError('No eligible project training backgrounds')
    data.update(items=records,parent_manifest_sha256=sha256(base/'dataset.json'),
                added_background_crops=len(added),background_review_required=True,
                counts_by_split={s:dict(Counter(o['class_name'] for r in records
                   if r['split']==s for o in r['objects'])) for s in ('train','validation')})
    (out/'dataset.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
    (out/'data.yaml').write_text(yaml.safe_dump({'path':str(out.resolve()),'train':'images/train',
                   'val':'images/validation','names':dict(enumerate(data['classes']))}),encoding='utf8')
    (out/'background_crops.json').write_text(json.dumps(added,indent=2)+'\n',encoding='utf8')
    columns=8;cell=144
    sheet=np.full((((len(added)+columns-1)//columns)*cell,columns*cell,3),40,np.uint8)
    for i,item in enumerate(added):
        im=read_image(out/item['image']);scale=min(138/im.shape[1],118/im.shape[0])
        resized=cv2.resize(im,None,fx=scale,fy=scale)
        x=(i%columns)*cell+3;y=(i//columns)*cell+22
        sheet[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
        cv2.putText(sheet,str(i),(x,y-5),0,.45,(0,255,0),1)
    write_image(out/'background_contact_sheet.png',sheet)
    audit=audit_dataset(out/'dataset.json')
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
    return {'status':'COMPILED_REVIEW_REQUIRED','backgrounds':len(added),'image_counts':audit['image_counts']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(compile_background(a.base,a.out)))
