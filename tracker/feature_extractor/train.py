import argparse
import os
import time
import warnings
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from net import EfficientNetClassifier
from datasets.market1501 import get_dataloader
from utils import increment_path


warnings.filterwarnings("ignore")
FILE = Path(__file__).absolute()


def main(opt):
    pre_weights = opt.pre_weights
    num_classes = opt.num_classes
    img_size = opt.img_size
    start_epoch = opt.start_epoch
    end_epoch = opt.end_epoch
    label_smoothing = opt.label_smoothing
    eval_interval = opt.eval_interval
    save_interval = opt.save_interval
    save_dir = opt.save_dir
    save_name = opt.save_name
    model_name = opt.model_name

    save_dir = increment_path(Path(save_dir) / save_name, exist_ok=False)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_save_path = os.path.join(save_dir, model_name + "_best.pt")
    last_save_path = os.path.join(save_dir, model_name + "_last.pt")
    log_save_path = os.path.join(save_dir, model_name + "_log.csv")

    model = EfficientNetClassifier(num_classes=num_classes)
    if pre_weights is not None:
        if os.path.isfile(pre_weights):
            model.load_param(pre_weights)
            print("Model is initialized with existing weights!")
        else:
            print("Model is initialized!")
    else:
        print("Model is initialized!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader, valid_loader = get_dataloader(img_size=img_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.99, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=1002, steps_per_epoch=1000)

    tmp_list = []
    for e in range(start_epoch, end_epoch + 1):
        print(f"\n###############  Epoch: {e:4} / {end_epoch:4}  ################")

        #time.sleep(0.5)
        train_loss = train(model, optimizer, train_loader, device)
        #time.sleep(0.5)

        if e % save_interval == 0 or e == end_epoch:
            torch.save(model.state_dict(), last_save_path)
        break


def train(model, optimizer, dataloader, device):
    model.train()
    mloss = np.array([0., 0., 0.])
    macc = 0.

    #for img0, img, pid, img_path in tqdm(dataloader):
    for img0, img, pid, img_path in dataloader:
        img = img.to(device)
        pid = pid.to(device)

        optimizer.zero_grad()
        score, feat = model(img)


def parse_opt():
    parser = argparse.ArgumentParser()

    pre_weights = None
    parser.add_argument("--pre-weights", type=str, default=pre_weights)
    parser.add_argument("--num-classes", type=int, default=1501)
    parser.add_argument("--img-size", type=int, default=[128, 64])
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--end-epoch", type=int, default=1000)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default=f"{FILE.parents[2]}/weights/feature_extractor")
    parser.add_argument("--save-name", type=str, default="exp")
    parser.add_argument("--model-name", type=str, default="feature_extractor_v1")

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
