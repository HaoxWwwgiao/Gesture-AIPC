"""Standalone frozen B image inference and metric projection; no camera or robot I/O."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from verify_runtime_assets import verify

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('MINIMUM_VISION_RUNTIME.json'))
    p.add_argument('--image',type=Path,required=True);p.add_argument('--same-camera-pose',action='store_true',required=True)
    p.add_argument('--backend',choices=['onnx_cpu','rocm'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();out=a.out.resolve()
    if out.exists():p.error('Fresh output directory required')
    inventory=json.loads(a.manifest.read_bytes());check=verify(root,a.manifest)
    if not check['integrity_passed']:raise ValueError('Runtime asset hash verification failed: '+json.dumps([r for r in check['results'] if r['status']!='MATCH']))
    opened=set()
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        try:
            target=Path(os.fsdecode(args[0])).resolve()
            if target.is_relative_to(root):opened.add(target.relative_to(root).as_posix())
        except (OSError,ValueError):pass
    sys.addaudithook(audit);sys.path[:0]=[str(root),str(root/'scripts')]
    os.environ.setdefault('MIOPEN_FIND_MODE','FAST')
    import numpy as np
    import torch
    from pc.mode_b.profile import Profile
    from pc.mode_b.perception import Perception
    from pc.mode_b.tiled_detector import TiledDetector
    from pc.mode_b.surface_ensemble import AlignedSurfaceEnsemble
    from pc.mode_ab.vision_workspace import WorkspaceDetector,project_rows
    from pc.rgb_table_geometry import RGBTableGeometry
    from pc.multiobject.dataset import read_image
    c=json.loads((root/inventory['configuration']).read_bytes())
    work=json.loads((root/c['workspace']).read_bytes())
    geometry=RGBTableGeometry(root/c['current_table'],root)
    frame=read_image(a.image);geometry.check_frame(frame.shape);torch.set_num_threads(4)
    if a.backend=='rocm':
        from pc.mode_ab.gpu import RocmDetector,RocmSurface
        detector=RocmDetector(root/inventory['detector_package']);surface=RocmSurface(root/inventory['surface_package'])
    else:
        from pc.multiobject.yolo import Detector
        from pc.multiobject.surface import SurfaceRuntime
        detector=Detector(root/inventory['detector_package'],'onnx_cpu');surface=SurfaceRuntime(root/inventory['surface_package'],'torch_cpu')
    projection_path=root/inventory['projection_source']
    spec=importlib.util.spec_from_file_location('frozen_metric_projection',projection_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    engine=Perception(WorkspaceDetector(TiledDetector(detector),geometry,c['localization_dimensions']),
        AlignedSurfaceEnsemble(surface),Profile.load(root/inventory['profile']))
    obs=engine.infer(frame,offline=True)
    xy=np.array(work['pattern_outer_corners_desktop_xyz_mm'])[:,:2];bounds=np.r_[xy.min(0),xy.max(0)]
    rows=project_rows(obs,geometry,c['localization_dimensions'],bounds,module.projected_top)
    report=dict(schema='gesture.aipc.standalone_frozen_image_replay.v1',backend=a.backend,
        input_image_sha256=hashlib.sha256(a.image.read_bytes()).hexdigest(),frame_sha256=obs['frame_sha256'],
        table_sha256=c['current_table_sha256'],coordinates='desk_global_mm',units='mm',
        localizations=rows,recognitions=obs['recognitions'],detector=obs['backend'],surface=surface.info,
        npu_execution_verified=False,independent_mm_accuracy_accepted=False,physical_execution=False,path_submissions=0,
        camera_access=False,robot_access=False,table_pose_assertion='OPERATOR_FLAG_NOT_NEW_MEASUREMENT',
        opened_runtime_files=sorted(opened))
    out.mkdir(parents=True);(out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='OFFLINE_INFERENCE_COMPLETED',backend=a.backend,
        recognitions=[r['class_name'] for r in obs['recognitions']],candidates=[dict(class_name=r['class_name'],status=r['status'],xyz=r['object_centre_desktop_xyz_mm']) for r in rows],
        physical_execution=False,path_submissions=0),ensure_ascii=False))

if __name__=='__main__':main()
