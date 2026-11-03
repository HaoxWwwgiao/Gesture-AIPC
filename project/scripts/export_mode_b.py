"""Static six-class raw YOLO export with explicit profile and source provenance."""
from pathlib import Path
import argparse
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pc.mode_b.profile import CLASSES,PROFILE_ID
from pc.multiobject.public_data import sha256
from pc.multiobject.yolo import load_package


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--side',type=int,help='Must match the static training input; defaults to its recorded size')
    a=p.parse_args()
    if a.out.exists():p.error('Fresh package required')
    import torch
    from ultralytics import YOLO,__version__
    torch.set_num_threads(4)
    report=json.loads((a.run/'training_report.json').read_bytes());weights=a.run/'weights/best.pt'
    trained_size=report['train_input_size_wh']
    if len(trained_size)!=2 or trained_size[0]!=trained_size[1]:p.error('Square training input required')
    if a.side is None:a.side=trained_size[0]
    if not 32<=a.side<=2048 or a.side%32 or trained_size!=[a.side,a.side]:
        p.error('Export size must match the recorded training input')
    model=YOLO(str(weights))
    if [model.names[i] for i in range(6)]!=CLASSES or report['classes']!=CLASSES:raise ValueError('Class order mismatch')
    out=model.export(format='onnx',imgsz=a.side,opset=17,dynamic=False,simplify=False,nms=False,batch=1,device='cpu')
    a.out.mkdir(parents=True);shutil.copyfile(out,a.out/'model.onnx')
    spec={'schema':'gesture.ai_pc.yolov8_package.v1','role':'mode_b_detector','profile_id':PROFILE_ID,
          'architecture':'yolov8s','classes':CLASSES,'trained_classes':report['trained_classes'],
          'input_size_wh':[a.side,a.side],'input_encoding':'RGB_FLOAT_0_1_LETTERBOX_114',
          'output_encoding':'YOLOV8_XYWH_CLASS_PROB_NO_NMS','precision':'fp32',
          'onnx':{'file':'model.onnx','sha256':sha256(a.out/'model.onnx')},
          'source_checkpoint_sha256':sha256(weights),'ultralytics_version':__version__,
          'license':'Ultralytics AGPL-3.0 / Enterprise; dataset notices recorded separately',
          'training_report':report,'metric_scope':report['metric_scope'],
          'class_quality_acceptance':'NOT_ACCEPTED_DEVELOPMENT_CANDIDATE',
          'npu_validation':'NOT_RUN','physical_execution':False}
    (a.out/'model.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf8');load_package(a.out/'model.json')
    print(json.dumps({'status':'EXPORTED','sha256':spec['onnx']['sha256']}),flush=True)


if __name__=='__main__':main()
