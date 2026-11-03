"""Compile reviewed public objects plus legacy train data into isolated mode B data."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse
import json
import sys
import cv2
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pc.multiobject.public_data import MODE_B_CLASSES,sha256,complete_scissors
from pc.multiobject.dataset import read_image,write_image


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Fresh dataset output required')
    base=ROOT/'artifacts/multiobject/mode_b_public_20261026'
    reviewed=ROOT/'docs/phase3c/mode_b_six_objects_20261026/public_annotations.json'
    anno=json.loads(reviewed.read_bytes())
    views={r['review_id']:r for r in json.loads((base/'annotation_views.json').read_bytes())}
    records=[];tiles=[];rejected=[];sources={};pixels={}
    def emit(image,objects,split,group,source,label_kind,foregrounds=None):
        # Reject identical pixels and preserve parent split for generated data.
        pixel=__import__('hashlib').sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        if pixel in pixels:
            if pixels[pixel]!=split:raise ValueError('Cross-split identical pixels')
            rejected.append({'reason':'DUPLICATE_PIXELS','source':source});return
        pixels[pixel]=split
        if group in sources and sources[group]!=split:raise ValueError('Source family crosses split')
        sources[group]=split
        index=len(records);name=f'item_{index:05d}'
        for kind in ('images','labels'):(a.out/kind/split).mkdir(parents=True,exist_ok=True)
        path=a.out/'images'/split/(name+'.png');write_image(path,image)
        h,w=image.shape[:2];lines=[]
        for o in objects:
            b=np.asarray(o['bbox_xyxy_raw'],float)
            if (b.shape!=(4,) or not np.isfinite(b).all() or np.any(b[:2]<0) or
                    b[2]>w or b[3]>h or np.any(b[2:]<=b[:2])):raise ValueError('Bad public annotation box')
            x,y,x2,y2=b;lines.append(f'{MODE_B_CLASSES.index(o["class_name"])} {(x+x2)/2/w:.9f} {(y+y2)/2/h:.9f} {(x2-x)/w:.9f} {(y2-y)/h:.9f}')
        (a.out/'labels'/split/(name+'.txt')).write_text('\n'.join(lines)+'\n',encoding='utf8')
        row={'image':path.relative_to(a.out).as_posix(),'image_sha256':sha256(path),
             'pixel_sha256':pixel,'image_size_wh':[w,h],'objects':objects,'split':split,
             'session_id':group,'source':source,'annotation_kind':label_kind}
        records.append(row)
        if split=='train' and foregrounds:
            for o,mask in zip(objects,foregrounds):
                x,y,x2,y2=np.rint(o['bbox_xyxy_raw']).astype(int)
                tiles.append({'image':image[y:y2,x:x2].copy(),'mask':mask[y:y2,x:x2].copy(),
                              'object':o,'source':source,'group':group,'offset':[x,y]})
    # Original cube train/validation assignments remain unchanged.
    old=ROOT/'artifacts/multiobject/exposure_field_20261025_01/prepared/dataset.json'
    data=json.loads(old.read_bytes())
    for row in data['items']:
        if row['split']=='test':continue
        objects=[o for o in row['objects'] if o['class_name'] in MODE_B_CLASSES]
        emit(read_image(old.parent/row['image']),objects,row['split'],row['session_id'],
             {'kind':'PROJECT_EXISTING_CUBE_DATA','manifest':old.relative_to(ROOT).as_posix(),'image_sha256':row['image_sha256']},row.get('annotation_kind','LEGACY'))
    # Office-Home provides class labels only. Boxes/surfaces here were separately
    # viewed and annotated by the assistant, never invented from the class label.
    for rid,label in anno['officehome'].items():
        row=views[rid];image=read_image(ROOT/row['local_image']);w,h=row['image_size_wh'];vw,vh=row['view_wh']
        scale=np.array([w/vw,h/vh]);objects=[];masks=[]
        for box in label.get('boxes',[label.get('box')]):
            b=(np.array(box).reshape(2,2)*scale).reshape(4).tolist()
            obj={'class_name':'eraser','bbox_xyxy_raw':b}
            if label.get('top'):obj['top_polygon_raw']=(np.array(label['top'])*scale).tolist()
            mask=np.zeros((h,w),np.uint8)
            if label.get('foreground'):
                cv2.fillPoly(mask,[np.rint(np.array(label['foreground'])*scale).astype(np.int32)],255)
            objects.append(obj);masks.append(mask)
        emit(image,objects,row['split'],'officehome_'+row['domain'].replace(' ','_')+'_'+Path(row['source_path']).name,
             {'kind':'OFFICE_HOME_PUBLIC','url':row['source_url'],'path':row['source_path'],'image_sha256':sha256(ROOT/row['local_image']),'source_license':'ACADEMIC_NONCOMMERCIAL_NOTICE'},
             'ASSISTANT_VISUAL_BOX_AND_SELECTED_TOP',masks if label.get('foreground') else None)
    for split,target_split in [('train','train'),('val','validation')]:
        public=json.loads((base/f'{split}_proposals.json').read_bytes())['records']
        accepted=anno['lvis'][('train' if split=='train' else 'validation')+'_canister_accepted_ids']
        by_image=defaultdict(list)
        for row in public:
            if row['class_name']=='scissors' or row['annotation']['id'] in accepted:by_image[row['image']['id']].append(row)
        full_index=json.loads((base/f'lvis_{split}_index.json').read_bytes())
        for iid,rows in by_image.items():
            by_image[iid]=complete_scissors(rows,full_index['annotations'][str(iid)])
        for iid,rows in by_image.items():
            image=read_image(ROOT/rows[0]['local_image']);h,w=image.shape[:2]
            # Whole annotated image, not per-instance unlabelled context crops.
            objects=[];masks=[]
            for row in rows:
                an=row['annotation'];x,y,bw,bh=an['bbox'];obj={'class_name':row['class_name'],'bbox_xyxy_raw':[x,y,min(w,x+bw),min(h,y+bh)]}
                rid='canister_train_'+str(an['id']);ellipse=anno['lvis']['canister_top_ellipses_in_views'].get(rid)
                if ellipse:
                    view=views[rid];cx,cy,rx,ry,angle=ellipse
                    poly=cv2.ellipse2Poly((int(cx),int(cy)),(int(rx),int(ry)),int(angle),0,360,8).astype(float)
                    crop=view['crop_xywh'];poly=poly*np.array(crop[2:])/view['view_wh']+crop[:2]
                    obj['top_polygon_raw']=poly.tolist()
                mask=np.zeros((h,w),np.uint8)
                for polygon in an.get('segmentation',[]):cv2.fillPoly(mask,[np.rint(np.array(polygon).reshape(-1,2)).astype(np.int32)],255)
                objects.append(obj);masks.append(mask)
            emit(image,objects,target_split,'lvis_'+split+'_'+str(iid),
                 {'kind':'LVIS_V1_PUBLIC','image_id':iid,'url':rows[0]['source_url'],'image_sha256':rows[0]['image_sha256'],'annotation_ids':[r['annotation']['id'] for r in rows]},
                 'LVIS_FULL_INSTANCE_BOX_MASK_SELECTED_TOP_AI_VISUAL',masks)
    # Train-only synthetic copy/paste augments reviewed source instances; it does
    # not create new independent samples or consume validation/test pixels.
    rng=np.random.default_rng(20261026)
    for index in range(96):
        tile=tiles[index%len(tiles)];im=tile['image'];mask=tile['mask'];hh,ww=im.shape[:2]
        if np.count_nonzero(mask)<50:continue
        side=448;scale=min(rng.uniform(110,265)/max(hh,ww),3)
        nw,nh=max(2,round(ww*scale)),max(2,round(hh*scale))
        rgb=cv2.resize(im,(nw,nh));m=cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_NEAREST)
        background=np.clip(rng.normal(rng.uniform(12,130),rng.uniform(2,12),(side,side,3)),0,255).astype(np.uint8)
        x,y=int(rng.integers(15,side-nw-15)),int(rng.integers(15,side-nh-15))
        region=background[y:y+nh,x:x+nw];region[m>0]=rgb[m>0]
        obj={'class_name':tile['object']['class_name'],'bbox_xyxy_raw':[x,y,x+nw,y+nh]}
        if tile['object'].get('top_polygon_raw'):
            points=(np.array(tile['object']['top_polygon_raw'])-tile['offset'])*np.array([nw/ww,nh/hh])+[x,y]
            obj['top_polygon_raw']=points.tolist()
        emit(background,[obj],'train',tile['group'],{'kind':'SYNTHETIC_COPY_PASTE_FROM_TRAIN_ONLY','parent':tile['source'],'independent_sample':False},'DERIVED_TRAIN_SOURCE_NOT_REAL_CAMERA')
    counts={s:dict(Counter(o['class_name'] for r in records if r['split']==s for o in r['objects'])) for s in ('train','validation')}
    manifest={'schema':'gesture.ai_pc.mode_b_dataset.v1','classes':MODE_B_CLASSES,'trained_classes':[n for n in MODE_B_CLASSES if counts['train'].get(n,0)>0],
              'items':records,'counts_by_split':counts,'source_split_policy':'UPSTREAM_SPLITS_AND_PROJECT_SESSIONS_PRESERVED_DERIVATIVES_TRAIN_ONLY',
              'public_annotation_file_sha256':sha256(reviewed),'test_used_for_training':False,'rejected_duplicates':rejected,
              'metric_scope':'DEVELOPMENT_PUBLIC_RELATED_OBJECTS_NOT_PROJECT_PHYSICAL_VALIDATION'}
    (a.out/'dataset.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    (a.out/'data.yaml').write_text(yaml.safe_dump({'path':str(a.out.resolve()),'train':'images/train','val':'images/validation','names':dict(enumerate(MODE_B_CLASSES))}),encoding='utf8')
    print(json.dumps({'images':len(records),'counts':counts,'tiles':len(tiles),'duplicates_skipped':len(rejected)}),flush=True)


if __name__=='__main__':main()
