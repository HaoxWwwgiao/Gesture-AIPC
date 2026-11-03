"""Check that angled training keeps source geometry, binary labels and negatives."""
from pathlib import Path
import sys
import unittest
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pc.mode_b.surface_augmentation import rotated_surface_sample


class SurfaceAugmentationTests(unittest.TestCase):
    def test_arbitrary_angles_keep_target_complete_and_aligned(self):
        image=np.zeros((300,500,3),np.uint8)
        polygon=np.array([[120,130],[360,130],[360,170],[120,170]],np.int32)
        cv2.fillPoly(image,[polygon],(0,0,255))
        for angle in [0,17,43,79,117,-153]:
            rgb,mask=rotated_surface_sample(image,[115,125,365,175],polygon,angle,.2)
            colour=(rgb[:,:,0]>.8)&(rgb[:,:,1]<.1)
            union=np.count_nonzero(colour|(mask>0))
            self.assertGreater(np.count_nonzero(colour&(mask>0))/union,.94)
            self.assertEqual(set(np.unique(mask)),{0,1})
            self.assertFalse(np.any(mask[[0,-1],:]))
            self.assertFalse(np.any(mask[:,[0,-1]]))

    def test_neighbour_stays_negative_and_source_is_unchanged(self):
        image=np.zeros((320,420,3),np.uint8)
        cv2.rectangle(image,(100,150),(300,180),(0,0,255),-1)
        cv2.circle(image,(220,120),12,(255,0,0),-1)
        original=image.copy()
        rgb,mask=rotated_surface_sample(image,[98,148,302,182],[[100,150],[300,150],[300,180],[100,180]],33,.3)
        blue=(rgb[:,:,2]>.8)&(rgb[:,:,0]<.1)
        self.assertGreater(blue.sum(),10)
        self.assertEqual(np.count_nonzero(mask[blue]),0)
        np.testing.assert_array_equal(image,original)

    def test_long_target_changes_roi_aspect_before_resize(self):
        image=np.zeros((300,500,3),np.uint8)
        cv2.rectangle(image,(100,140),(400,160),(0,0,255),-1)
        _,mask=rotated_surface_sample(image,[100,140,400,160],[[100,140],[400,140],[400,160],[100,160]],45,.2)
        ys,xs=np.where(mask>0)
        self.assertLess(abs((xs.max()-xs.min())-(ys.max()-ys.min())),4)
        self.assertGreater(xs.max()-xs.min(),100)

    def test_detector_box_jitter_keeps_pixel_labels_aligned(self):
        from train_mode_b_surface import SurfaceSamples
        image=np.zeros((300,500,3),np.uint8)
        polygon=[[120,130],[360,130],[360,170],[120,170]]
        cv2.fillPoly(image,[np.array(polygon,np.int32)],(0,0,255))
        samples=SurfaceSamples.__new__(SurfaceSamples)
        row={'image':'fixture'}
        obj={'bbox_xyxy_raw':[115,125,365,175],'top_polygon_raw':polygon}
        samples.cache={'fixture':image}
        samples.samples={'eraser':[(row,obj)]}
        xs,ys=samples.batch(np.random.default_rng(55),12,continuous_rotation=True,roi_jitter=.04)
        for rgb,label in zip(xs.numpy(),ys.numpy()):
            colour=(rgb[0]>.6)&(rgb[1]<.1)
            target=label[0]>.5
            self.assertGreater(np.count_nonzero(colour&target)/np.count_nonzero(colour|target),.94)
            self.assertFalse(target[[0,-1],:].any())
            self.assertFalse(target[:,[0,-1]].any())


if __name__=='__main__':unittest.main(verbosity=2)
