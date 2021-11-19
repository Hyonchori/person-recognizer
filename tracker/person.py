# Person class detected in image

import os
import sys
from pathlib import Path

import cv2
from PIL import Image
import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(os.path.join(str(ROOT)))

if not os.path.join(str(ROOT), "yolov5") in sys.path:
    sys.path.append(os.path.join(str(ROOT), "yolov5"))
from yolov5.utils.datasets import IMG_FORMATS


# Temporal person class for person detected in current image
class Person():
    def __init__(self, person_id, query_path, feature_extractor, transform, device, img_resize=(128, 256)):
        self.person_id = person_id
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.device = device
        self.img_resize = img_resize

        self.query_path = query_path
        self.query_img_files = [x for x in sorted(os.listdir(query_path)) if x.split(".")[-1] in IMG_FORMATS]
        self.query_img_batch = self.get_img_batch(self.query_img_files)
        self.query_feat_batch = self.get_feat_batch(self.query_img_batch)

    def get_img_batch(self, img_files):
        img_batch = []
        for img_file in img_files:
            img_path = os.path.join(self.query_path, img_file)
            img = np.array(Image.open(img_path).convert('RGB'))
            img = cv2.resize(img, dsize=self.img_resize)
            img = self.transform(img)[None]
            img_batch.append(img)
        img_batch = torch.cat(img_batch)
        print(f"person id {self.person_id}'s query images are loaded '{img_batch.shape}'")
        return img_batch

    def get_feat_batch(self, img_batch):
        img_batch = img_batch.to(self.device).type_as(next(self.feature_extractor.parameters()))
        with torch.no_grad():
            feats = self.feature_extractor(img_batch).detach()
        return feats

