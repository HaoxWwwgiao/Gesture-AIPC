"""Warm-start six classes from proven cube features and official COCO scissors."""
from pathlib import Path
import argparse
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pc.multiobject.public_data import MODE_B_CLASSES,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=35)
    p.add_argument('--side',type=int,default=384)
    p.add_argument('--warm-start',type=Path)
    a=p.parse_args()
    if a.out.exists() or a.side%32:p.error('Fresh model output and static multiple-of-32 side required')
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(6);torch.manual_seed(20261026)
    data=json.loads((a.dataset/'dataset.json').read_bytes())
    if data.get('background_review_required'):
        p.error('Background contact-sheet review must be recorded before training')
    if data['classes']!=MODE_B_CLASSES or data['trained_classes']!=MODE_B_CLASSES:
        p.error('Six classes with training positives required')
    cube=ROOT/'artifacts/multiobject/exposure_field_20261025_01/train_detector/weights/best.pt'
    coco=ROOT/'artifacts/multiobject/source/yolov8s.pt'
    copied=[];initial=None
    if a.warm_start:
        model=YOLO(str(a.warm_start))
        if list(model.names.values())!=MODE_B_CLASSES:raise ValueError('Warm start six-class order mismatch')
    else:
        model=initialise_head(cube,coco,copied)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        initial=a.out.parent/(a.out.name+'_initial.pt')
        if initial.exists():raise ValueError('Initial checkpoint already exists')
        model.save(str(initial));model=YOLO(str(initial))
    start=time.perf_counter()
    model.train(data=str((a.dataset/'data.yaml').resolve()),epochs=a.epochs,imgsz=a.side,batch=4 if a.side>384 else 8,
                device='cpu',workers=0,project=str(a.out.parent.resolve()),name=a.out.name,exist_ok=False,
                seed=20261026,deterministic=True,amp=False,plots=False,freeze=10,
                hsv_h=0.,hsv_s=.10,hsv_v=.25,mosaic=.35,close_mosaic=5,mixup=0.,
                fliplr=.5,flipud=.2,degrees=30.,scale=.25,translate=.15,patience=12,
                optimizer='AdamW',lr0=.00012 if a.warm_start else .00035)
    field_counts={name:sum(o['class_name']==name for r in data['items'] if r['split']=='train'
                          and r['source']['kind'] in ('PROJECT_MEDICINE_BOTTLE_FIELD','PROJECT_SIX_OBJECT_FIELD')
                          for o in r['objects']) for name in MODE_B_CLASSES}
    report={'schema':'gesture.ai_pc.mode_b_training.v1','classes':MODE_B_CLASSES,
            'trained_classes':MODE_B_CLASSES,'dataset_sha256':sha256(a.dataset/'dataset.json'),
            'dataset_counts':data['counts_by_split'],'cube_checkpoint_sha256':sha256(cube),
            'official_coco_checkpoint_sha256':sha256(coco),'initial_checkpoint_sha256':sha256(initial) if initial else None,
            'head_transfer':copied,'epochs_requested':a.epochs,'elapsed_s':time.perf_counter()-start,
            'train_input_size_wh':[a.side,a.side],
            'warm_start_sha256':sha256(a.warm_start) if a.warm_start else None,
            'device':'cpu','torch_version':torch.__version__,'test_used_for_training':False,
            'validation_used_for_model_selection':True,
            'metric_scope':'PUBLIC_AND_PROJECT_DEVELOPMENT_ONLY',
            'project_eraser_canister_photos_available':any(field_counts.values()),
            'project_field_original_instances_by_class':field_counts,'physical_execution':False,
            'npu_acceptance':'NOT_ACCEPTED','rocm_acceptance':'NOT_ACCEPTED'}
    (a.out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':'TRAINING_COMPLETED','out':str(a.out),'elapsed_s':report['elapsed_s']}),flush=True)


def initialise_head(cube,coco,copied):
    from ultralytics import YOLO
    import torch
    from torch import nn
    model=YOLO(str(cube));reference=YOLO(str(coco));head=model.model.model[-1];ref_head=reference.model.model[-1]
    # The final six-channel classifier retains learned RGB channels 0/1/2.
    # Both YOLOv8s heads use 128 hidden channels. COCO class76 supplies scissors;
    # it remains only a warm start under the field-adapted shared features.
    for scale,branch in enumerate(head.cv3):
        old=branch[-1];new=nn.Conv2d(old.in_channels,6,1)
        with torch.no_grad():
            new.bias.fill_(-5.)
            new.weight[:3].copy_(old.weight[:3]);new.bias[:3].copy_(old.bias[:3])
            new.weight[5].copy_(ref_head.cv3[scale][-1].weight[76])
            new.bias[5].copy_(ref_head.cv3[scale][-1].bias[76])
        branch[-1]=new;copied.append({'scale':scale,'cube_rows':[0,1,2],'coco_scissors_row':76,'new_rows':[3,4]})
    head.nc=6;head.no=head.reg_max*4+6
    model.model.yaml['nc']=6;model.model.names=dict(enumerate(MODE_B_CLASSES))
    model.overrides['model']=str(cube)
    return model


if __name__=='__main__':main()
