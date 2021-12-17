import argparse
import os
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import albumentations as at
from tqdm import tqdm

from net import EfficientNetClassifier
from metrics import euclidean_distance, cosine_similarity
from utils import increment_path

warnings.filterwarnings("ignore")
FILE = Path(__file__).absolute()


def get_query_imgs(img_dir, select_query):
    img_dict = {}
    for img_name in os.listdir(img_dir):
        if not img_name.endswith(".jpg"):
            continue
        img_id = int(img_name.split("_")[0])
        if img_id in img_dict:
            if len(img_dict[img_id]) < select_query:
                img_dict[img_id].append(img_name)
            else:
                continue
        else:
            img_dict[img_id] = [img_name]
    query_imgs = []
    for v in img_dict.values():
        query_imgs += v
    return query_imgs


@torch.no_grad()
def main(opt):
    weights = opt.weights
    num_classes = opt.num_classes
    img_size = opt.img_size
    query_path = opt.query_path
    gallery_path = opt.gallery_path
    gallery_save_path = opt.gallery_save_path
    select_query = opt.select_query
    rank_num = opt.rank_num
    save_dir = opt.save_dir
    save_name = opt.save_name
    save = opt.save

    save_dir = increment_path(Path(save_dir) / save_name, exist_ok=False)
    if save:
        save_dir.mkdir(parents=True, exist_ok=True)

    model = EfficientNetClassifier(num_classes=num_classes)
    if weights is not None:
        if os.path.isfile(weights):
            model.load_state_dict(torch.load(weights))
            print("Model is initialized with existing weights!")
        else:
            print("Model is initialized!")
    else:
        print("Model is initialized!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    cudnn.benchmark = True

    query_imgs = get_query_imgs(query_path, select_query)
    gallery_imgs = [x for x in os.listdir(gallery_path) if x.endswith(".jpg")]
    transform = at.Compose([
        at.Resize(*img_size),
        at.Normalize()
    ])
    if os.path.isfile(gallery_save_path):
        gallery_feats = torch.load(gallery_save_path)
        print("\n--- Load gallery feature ...")
    else:
        print(f"\n--- Given gallery_save_path '{gallery_save_path}' is wrong ...")
        gallery_feats = None
    if gallery_feats is None:
        gallery_feats = []
        print("\n--- Make gallery feature ...")
        time.sleep(0.5)
        for gallery_img_name in tqdm(gallery_imgs):
            img_path = os.path.join(gallery_path, gallery_img_name)
            img = cv2.imread(img_path)
            if transform is not None:
                transformed = transform(image=img)
                img = transformed["image"]
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)[None].to(device)
            score, feat = model(img)
            gallery_feats.append(feat)
        gallery_feats = torch.cat(gallery_feats)
        gallery_save_path = os.path.join(save_dir, "gallery_feats.pt")
        if save:
            torch.save(gallery_feats, gallery_save_path)
    print(gallery_feats.shape)

    print("\n--- Make visualization ...")
    time.sleep(0.5)
    for query_img_name in tqdm(query_imgs):
        img_path = os.path.join(query_path, query_img_name)
        img0 = cv2.imread(img_path)
        if transform is not None:
            transformed = transform(image=img0)
            img = transformed["image"]
        else:
            img = img0
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)[None].to(device)
        score, feat = model(img)
        dist_mat = euclidean_distance(feat, gallery_feats)
        indices = np.argsort(dist_mat, axis=1)[0]
        visualize(img0, indices, gallery_path, gallery_imgs, save_dir, save=save, rank_num=rank_num)


def visualize(img, indices, gallery_path, gallery_imgs, save_dir, save=True, rank_num=10):
    similar_imgs = []
    for i in range(rank_num):
        gallery_img_name = gallery_imgs[indices[i]]
        gallery_img_path = os.path.join(gallery_path, gallery_img_name)
        gallery_img = cv2.imread(gallery_img_path)
        similar_imgs.append(gallery_img)
    similar_imgs = np.hstack(similar_imgs)
    black_padding = np.zeros_like(img)
    vis = np.hstack((img, black_padding, similar_imgs))
    cv2.imshow("vis", vis)
    cv2.waitKey(0)

def parse_opt():
    parser = argparse.ArgumentParser()

    weights = f"{FILE.parents[2]}/weights/feature_extractor/exp1/feature_extractor_v1_last.pt"
    parser.add_argument("--weights", type=str, default=weights)
    parser.add_argument("--num-classes", type=str, default=1501)
    parser.add_argument("--img-size", type=int, default=[128, 64])

    query_path = "/media/daton/D6A88B27A88B0569/dataset/market1501/Custom_dataset/valid"
    parser.add_argument("--query-path", type=str, default=query_path)

    gallery_path = "/media/daton/D6A88B27A88B0569/dataset/market1501/Custom_dataset/train"
    parser.add_argument("--gallery-path", type=str, default=gallery_path)
    gallery_save_path = f"{FILE.parents[0]}/inference/exp34/gallery_feats.pt"
    parser.add_argument("--gallery-save-path", type=str, default=gallery_save_path)

    parser.add_argument("--select-query", type=int, default=5)
    parser.add_argument("--rank-num", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default=f"{FILE.parents[0]}/inference")
    parser.add_argument("--save-name", type=str, default="exp")
    parser.add_argument("--save", action="store_true", default=True)

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
