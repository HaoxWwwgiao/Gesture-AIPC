"""Expand existing surface weights using explicit TRAIN-only visible top polygons."""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import sys
import time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pc.multiobject.dataset import read_image
from pc.multiobject.public_data import sha256


class SurfaceSamples:
    def __init__(self,path):
        self.path=Path(path);self.data=json.loads(self.path.read_bytes())
        self.samples=defaultdict(list);self.cache={};sessions={};hashes={}
        for row in self.data['items']:
            group,split=row['session_id'],row['split']
            if group in sessions and sessions[group]!=split:raise ValueError('Source split leakage')
            sessions[group]=split
            h=row['image_sha256']
            if h in hashes and hashes[h]!=split:raise ValueError('Image split leakage')
            hashes[h]=split
            if split!='train':continue
            for obj in row['objects']:
                if not obj.get('top_polygon_raw'):continue
                image=read_image(self.path.parent/row['image']);hh,ww=image.shape[:2]
                poly=np.array(obj['top_polygon_raw'],float)
                if (poly.ndim!=2 or poly.shape[1]!=2 or len(poly)<4 or not np.isfinite(poly).all() or
                        np.any(poly<0) or np.any(poly>[ww-1,hh-1])):raise ValueError('Bad explicit top polygon')
                self.cache[row['image']]=image
                self.samples[obj['class_name']].append((row,obj))
        if not {'red_cube','eraser','small_canister'}.issubset(self.samples):raise ValueError('All three grasp target surface examples needed')

    def batch(self,rng,n,side=256,continuous_rotation=False,roi_jitter=0.):
        import torch
        xs=[];ys=[];names=sorted(self.samples)
        for _ in range(n):
            name=names[int(rng.integers(len(names)))];row,obj=self.samples[name][int(rng.integers(len(self.samples[name])))]
            image=self.cache[row['image']];h,w=image.shape[:2];box=np.array(obj['bbox_xyxy_raw'],float)
            if roi_jitter:
                # Detector boxes move slightly even on a static object. Keep
                # image/label pixels fixed while perturbing only the input ROI.
                span=box[2:]-box[:2]
                shift=rng.uniform(-roi_jitter,roi_jitter,2)*span
                scale=np.exp(rng.uniform(-roi_jitter,roi_jitter,2))
                centre=(box[:2]+box[2:])/2+shift
                box=np.r_[centre-span*scale/2,centre+span*scale/2]
            if continuous_rotation:
                from pc.mode_b.surface_augmentation import rotated_surface_sample
                angle=float(rng.uniform(-180,180)) if rng.random()<.75 else 0.
                rgb,target=rotated_surface_sample(image,box,obj['top_polygon_raw'],angle,rng.uniform(.15,.30),side)
            else:
                pad=max(box[2:]-box[:2])*rng.uniform(.15,.30)
                x,y=np.maximum(np.floor(box[:2]-pad),0).astype(int);x2,y2=np.minimum(np.ceil(box[2:]+pad),[w,h]).astype(int)
                mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[np.rint(obj['top_polygon_raw']).astype(np.int32)],1)
                rgb=cv2.cvtColor(cv2.resize(image[y:y2,x:x2],(side,side)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
                target=cv2.resize(mask[y:y2,x:x2],(side,side),interpolation=cv2.INTER_NEAREST)
            rgb=np.clip(rgb*rng.uniform(.75,1.2),0,1)
            turns=int(rng.integers(4));rgb,target=np.rot90(rgb,turns),np.rot90(target,turns)
            if rng.random()<.5:rgb,target=rgb[:,::-1],target[:,::-1]
            xs.append(np.ascontiguousarray(rgb.transpose(2,0,1)));ys.append(np.ascontiguousarray(target[None]))
        return torch.from_numpy(np.stack(xs)),torch.from_numpy(np.stack(ys)).float()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=300)
    p.add_argument('--warm-start',type=Path,help='Surface model.json; default retains original RGB warm start')
    p.add_argument('--continuous-rotation',action='store_true',help='Rigid scene rotation before recropping the target')
    p.add_argument('--roi-jitter',type=float,default=0.,help='Train-only relative detector-box shift/scale, 0 to 0.15')
    p.add_argument('--positive-weight',type=float,default=2.,help='Foreground BCE weight; original behavior is 2')
    a=p.parse_args()
    if not 0<=a.roi_jitter<=.15 or not .25<=a.positive_weight<=4:
        p.error('ROI jitter must be 0..0.15 and positive weight 0.25..4')
    if a.out.exists():p.error('Fresh surface package required')
    import torch
    from torch.nn import functional as F
    from pc.multiobject.surface import SurfaceNet
    torch.set_num_threads(4);torch.manual_seed(20261026);rng=np.random.default_rng(20261026)
    ds=SurfaceSamples(a.dataset/'dataset.json')
    weights=ROOT/'model_package/multiobject/surface_exposure_field_v1/weights.pt'
    if a.warm_start:
        warm=json.loads(a.warm_start.read_bytes())
        weights=a.warm_start.parent/warm['checkpoint']['file']
        if warm['architecture']!=SurfaceNet.architecture or sha256(weights)!=warm['checkpoint']['sha256']:
            raise ValueError('Surface warm start architecture/hash mismatch')
    model=SurfaceNet();model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    for p in model.encoder.parameters():p.requires_grad=False
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003)
    a.out.mkdir(parents=True);start=time.perf_counter()
    with (a.out/'training.jsonl').open('w',encoding='utf8') as log:
        for step in range(1,a.steps+1):
            model.train();model.encoder.eval();x,y=ds.batch(rng,6,continuous_rotation=a.continuous_rotation,roi_jitter=a.roi_jitter)
            logits=model(x);prob=logits.sigmoid()
            loss=F.binary_cross_entropy_with_logits(logits,y,pos_weight=torch.tensor(a.positive_weight))
            loss=loss+.5*(1-(2*(prob*y).sum()+1)/(prob.sum()+y.sum()+1))
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            row={'step':step,'loss':float(loss.detach()),'elapsed_s':time.perf_counter()-start}
            log.write(json.dumps(row)+'\n')
            if step%50==0:print(json.dumps(row),flush=True)
    model.eval();torch.save(model.state_dict(),a.out/'weights.pt')
    spec={'schema':'gesture.ai_pc.surface_package.v1','architecture':SurfaceNet.architecture,
          'input_encoding':'RGB_FLOAT_0_1_INTERNAL_IMAGENET_NORMALIZATION','input_side':256,
          'checkpoint':{'file':'weights.pt','sha256':sha256(a.out/'weights.pt')},
          'warm_start_sha256':sha256(weights),'dataset_sha256':sha256(a.dataset/'dataset.json'),
          'encoder_frozen':True,'steps':a.steps,'batch':6,'class_balanced':True,
          'detector_roi_jitter_fraction':a.roi_jitter,'foreground_bce_positive_weight':a.positive_weight,
          'spatial_augmentation':('RIGID_SCENE_ROTATION_75_PERCENT_UNIFORM_MINUS180_PLUS180_BEFORE_ROI_RESIZE'
                                  if a.continuous_rotation else 'ROT90_AND_HORIZONTAL_FLIP_AFTER_ROI_RESIZE'),
          'training_sample_counts_including_derivatives':{k:len(v) for k,v in ds.samples.items()},
          'trained_surface_classes':sorted(ds.samples),
          'training_labels':'ASSISTANT_SELECTED_VISIBLE_TOP_POLYGONS_PUBLIC_AND_PROJECT_TRAIN_ONLY',
          'top_surface_gravity_not_measured_for_public_images':True,
          'validation_top_classes_available':['eraser'],
          'canister_independent_surface_validation':'NOT_AVAILABLE',
          'metric_scope':'DEVELOPMENT_ONLY_SMALL_PUBLIC_TOP_SAMPLE_COVERAGE',
          'training_device':'CPU','rocm_validation':'NOT_RUN','physical_execution':False,
          'elapsed_s':time.perf_counter()-start}
    (a.out/'model.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':'SURFACE_TRAINED','out':str(a.out),'elapsed_s':spec['elapsed_s']}),flush=True)


if __name__=='__main__':main()
