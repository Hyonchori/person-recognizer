# Person class detected in image

import os
import sys
from pathlib import Path

import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(os.path.join(str(ROOT)))
sys.path.append(os.path.join(str(ROOT), "yolov5"))
from yolov5.utils.general import xyxy2xywh


# Temporal person class for person detected in current image
class Person():
    def __init__(self, ref_img, box, img_resize=(128, 256)):
        self.img = cv2.resize(ref_img[int(box[1]): int(box[3]), int(box[0]): int(box[2])], dsize=img_resize)
        self.xyxy = box
        self.cpwh = xyxy2xywh(box[None])