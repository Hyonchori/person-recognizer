import os

import cv2
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def split_dataset(root, target_dir, out_dir, dir_name1, dir_name2, split_size=0.4, save=True):
    img_dir = os.path.join(root, target_dir)
    img_names = [x for x in os.listdir(img_dir) if x.endswith(".jpg")]
    img_ids = [int(x.split("_")[0]) for x in img_names]
    split = StratifiedShuffleSplit(n_splits=1, test_size=split_size, random_state=42)
    for train_idx, valid_idx in split.split(img_names, img_ids):
        train_imgs = np.array(img_names)[train_idx]
        valid_imgs = np.array(img_names)[valid_idx]
    if save:
        make_dataset(train_imgs, img_dir, os.path.join(root, out_dir), dir_name1)
        make_dataset(valid_imgs, img_dir, os.path.join(root, out_dir), dir_name2)


def make_dataset(img_names, img_dir, out_dir, dir_name):
    print(f"\n--- Make dataset '{dir_name}'")
    out_dir_path = os.path.join(out_dir, dir_name)
    if not os.path.exists(out_dir_path) or not os.path.isdir(out_dir_path):
        os.makedirs(out_dir_path)
    for img_name in tqdm(img_names):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        out_path = os.path.join(out_dir_path, img_name)
        cv2.imwrite(out_path, img)


if __name__ == "__main__":
    root = "/media/daton/D6A88B27A88B0569/dataset/market1501"
    target_dir = "gt_bbox"
    out_dir = "Custom_dataset/query_gallery"

    split_dataset(root, target_dir, out_dir, dir_name1="gallery", dir_name2="query")
