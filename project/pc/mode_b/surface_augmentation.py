"""Training-only rigid rotations before the target ROI is resized."""
import cv2
import numpy as np


def rotated_surface_sample(image, box, polygon, angle_deg, pad_fraction, side=256):
    """Rotate scene and annotation together, then recrop the transformed box.

    Applying rotation before ROI resizing reproduces the changing axis-aligned
    crop of a long object. Rotating an already resized square does not do this.
    The target mask stays binary and neighbouring surfaces remain negatives.
    """
    box=np.asarray(box,dtype=float)
    polygon=np.asarray(polygon,dtype=float)
    if (image.ndim!=3 or image.dtype!=np.uint8 or box.shape!=(4,)
            or np.any(box[2:]<=box[:2]) or not np.isfinite(box).all()
            or polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<3
            or not np.isfinite(polygon).all() or not np.isfinite(angle_deg)
            or not .05<=pad_fraction<=.5 or side<32):
        raise ValueError('Valid image, box, polygon, rotation and padding required')
    centre=(box[:2]+box[2:])/2
    # A complete rotation of the target and its context fits inside this square.
    radius=int(np.ceil(np.linalg.norm(box[2:]-box[:2])*(.5+pad_fraction)))+4
    origin=np.floor(centre-radius).astype(int)
    length=radius*2+1
    context=np.zeros((length,length,3),np.uint8)
    h,w=image.shape[:2]
    lo=np.maximum(origin,0);hi=np.minimum(origin+length,[w,h])
    if np.any(hi<=lo):raise ValueError('Target lies outside image')
    dest=lo-origin
    context[dest[1]:dest[1]+hi[1]-lo[1],dest[0]:dest[0]+hi[0]-lo[0]]=image[lo[1]:hi[1],lo[0]:hi[0]]
    mask=np.zeros((length,length),np.uint8)
    cv2.fillPoly(mask,[np.rint(polygon-origin).astype(np.int32)],1)
    transform=cv2.getRotationMatrix2D(tuple(centre-origin),float(angle_deg),1.)
    warped=cv2.warpAffine(context,transform,(length,length),flags=cv2.INTER_LINEAR)
    target=cv2.warpAffine(mask,transform,(length,length),flags=cv2.INTER_NEAREST)
    x,y,x2,y2=box
    corners=np.array([[x,y],[x2,y],[x2,y2],[x,y2]])-origin
    corners=np.column_stack((corners,np.ones(4)))@transform.T
    low,high=corners.min(0),corners.max(0)
    pad=max(high-low)*pad_fraction
    low=np.maximum(np.floor(low-pad),0).astype(int)
    high=np.minimum(np.ceil(high+pad),length).astype(int)
    if np.any(high<=low):raise ValueError('Empty augmented crop')
    rgb=cv2.cvtColor(cv2.resize(warped[low[1]:high[1],low[0]:high[0]],(side,side)),cv2.COLOR_BGR2RGB)
    target=cv2.resize(target[low[1]:high[1],low[0]:high[0]],(side,side),interpolation=cv2.INTER_NEAREST)
    return rgb.astype(np.float32)/255.,target
