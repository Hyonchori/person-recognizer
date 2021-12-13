import os
import math
import random
from collections import Counter

import torch
import numpy as np
from PIL import Image
import albumentations as at


class MARKET1501Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 root: str,
                 transform: at.core.composition.Compose=None):
        self.img_dir = os.path.join(root, "gt_bbox")
        self.img_names = [x for x in os.listdir(self.img_dir) if x.endswith(".jpg")]
        self.id_counter = Counter([int(x.split("_")[0]) for x in self.img_names])
        self.ids = list(self.id_counter.keys())
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, x):
        img_name = self.img_names[x]
        name_split = img_name.split("_")
        pid = int(name_split[0])
        img_path = os.path.join(self.img_dir, img_name)
        img0 = cv2.imread(img_path)

        if self.transform is not None:
            transformed = self.transform(image=img0)
            img = transformed["image"]
        else:
            img = img0
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        return img0, img, pid, img_path


def get_dataloader(root: str,
                   img_size: (int, int)=(128, 64),
                   train_batch: int=32,
                   valid_batch: int=32):
    train_transform = at.Compose([
        at.Resize(*img_size),
        at.Rotate(limit=(-20, 20)),
        at.Affine(translate_percent=(-0.2, 0.2)),
        at.Perspective(),
        at.HorizontalFlip(),
        at.CoarseDropout(1, 32, 48),
        at.Normalize()
    ])
    valid_transform = at.Compose([
        at.Resize(128, 64),
        at.Normalize()
    ])
    root_dataset = MARKET1501Dataset(root, train_transform)
    
    train_loader = torch.utils.data.DataLoader(
        root_dataset,
        batch_size=train_batch,
        shuffle=True,
    )
    return train_loader


if __name__ == "__main__":
    import cv2

    root = "/media/daton/D6A88B27A88B0569/dataset/market1501"
    train_transform = at.Compose([
        at.Resize(224, 112),
        at.Rotate(limit=(-20, 20)),
        at.Affine(translate_percent=(-0.2, 0.2)),
        at.Perspective(),
        at.HorizontalFlip(),
        at.CoarseDropout(1, 32, 48),
        at.Normalize()
    ])
    # Dataset test
    '''dataset = MARKET1501Dataset(root, transform=train_transform)
    for img0, img, pid, img_path in dataset:
        print("\n---")
        print(img0.shape, img.shape)
        cv2.imshow("img0", img0)
        cv2.imshow("img", img.transpose(1, 2, 0)[..., ::-1])
        cv2.waitKey(0)
        pass'''

    # Dataloader test
    trainloader = get_dataloader(root)
    for img0, img, pid, img_path in trainloader:
        print("\n---")
        print(img0.shape)
        print(img.shape)
        print(pid.shape)
        print(pid)
        break
