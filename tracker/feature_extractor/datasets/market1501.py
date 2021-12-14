import os
from collections import Counter

import torch
from torch.utils.data import Subset
import cv2
import numpy as np
import albumentations as at


class MARKET1501Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 root: str,
                 img_dir: str,
                 transform: at.core.composition.Compose=None):
        self.img_dir = os.path.join(root, img_dir)
        self.img_names = [x for x in os.listdir(self.img_dir) if x.endswith(".jpg")]
        self.transform = transform

    def get_labels(self):
        return [int(x.split("_")[0]) for x in self.img_names]

    def get_label_count(self):
        id_counter = Counter([int(x.split("_")[0]) for x in self.img_names])
        return id_counter

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


root = "/media/daton/D6A88B27A88B0569/dataset/market1501"
train_dir = "Custom_dataset/train"
valid_dir = "Custom_dataset/valid"


def get_dataloader(root: str = root,
                   train_dir: str = train_dir,
                   valid_dir: str = valid_dir,
                   img_size: (int, int) = (128, 64),
                   train_batch: int = 32,
                   valid_batch: int = 32):
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
        at.Resize(*img_size),
        at.Normalize()
    ])
    train_dataset = MARKET1501Dataset(root, train_dir, transform=train_transform)
    valid_dataset = MARKET1501Dataset(root, valid_dir, transform=valid_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_batch,
        shuffle=False
    )
    return train_loader, valid_loader


if __name__ == "__main__":
    import cv2

    root = "/media/daton/D6A88B27A88B0569/dataset/market1501"
    train_dir = "Custom_dataset/train"
    valid_dir = "Custom_dataset/valid"
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
    train_loader, valid_loader = get_dataloader(root, train_dir, valid_dir)
    for img0, img, pid, img_path in valid_loader:
        print("\n---")
        print(img0.shape)
        print(img.shape)
        print(pid.shape)
        print(pid)
        for im0, im, id in zip(img0, img, pid):
            print(id.numpy())
            im = im.numpy().transpose(1, 2, 0)[..., ::-1]
            cv2.imshow("im0", im0.numpy())
            cv2.imshow("im", im)
            cv2.waitKey(0)
        break
