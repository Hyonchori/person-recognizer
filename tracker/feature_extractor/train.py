import argparse
import copy
import os
import csv
import time
import warnings
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from net import EfficientNetClassifier
from datasets.market1501 import get_dataloader
from losses import ComputeLoss
from utils import increment_path


warnings.filterwarnings("ignore")
FILE = Path(__file__).absolute()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(opt):
    pre_weights = opt.pre_weights
    num_classes = opt.num_classes
    img_size = opt.img_size
    start_epoch = opt.start_epoch
    end_epoch = opt.end_epoch
    batch_size = opt.batch_size
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
    params = list(model.named_parameters())
    last_feature_dim = params[-2][1].shape[0]
    model = model.to(device)

    train_loader, valid_loader = get_dataloader(img_size=img_size, train_batch=batch_size)
    compute_loss = ComputeLoss(label_smoothing=label_smoothing,
                               num_classes=num_classes,
                               last_feature_dim=last_feature_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    optimizer_center = torch.optim.AdamW(compute_loss.cnt_loss_fn.parameters(), lr=0.003, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 500, 700, 900], gamma=0.5)

    best_loss = 100000.

    for e in range(start_epoch, end_epoch + 1):
        print(f"\n###############  Epoch: {e:4} / {end_epoch:4}  ################")

        time.sleep(0.5)
        train_loss, train_acc = train(model, compute_loss, optimizer, optimizer_center, train_loader, device)
        time.sleep(0.5)
        print(f"train loss: \n\tcls: {train_loss[0]:.4f}, triplet: {train_loss[1]:.4f}, center: {train_loss[2]:.4f}, total: {sum(train_loss):.4f}")
        print(f"\ttrain accuracy: {train_acc:.6f}")

        if e % eval_interval == 0:
            time.sleep(0.5)
            valid_loss, valid_acc = evaluate(model, compute_loss, valid_loader, device)
            time.sleep(0.5)
            print(f"valid loss: \n\tcls: {valid_loss[0]:.4f}, triplet: {valid_loss[1]:.4f}, center: {valid_loss[2]:.4f}, total: {sum(valid_loss):.4f}")
            print(f"\tvalid accuracy: {valid_acc:.6f}")

        if e % save_interval == 0 or e == end_epoch:
            torch.save(model.state_dict(), last_save_path)

        if os.path.isfile(log_save_path):
            with open(log_save_path, "r") as f:
                reader = csv.reader(f)
                logs = list(reader)
                logs.append([e] +
                            [x for x in train_loss] +
                            [x for x in valid_loss] +
                            [optimizer.param_groups[0]["lr"]])
            with open(log_save_path, "w") as f:
                writer = csv.writer(f)
                for log in logs:
                    writer.writerow(log)
        else:
            with open(log_save_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow([e] +
                                [x for x in train_loss] +
                                [x for x in valid_loss] +
                                [optimizer.param_groups[0]["lr"]])

        if sum(valid_loss) < best_loss:
            best_loss = sum(valid_loss)
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, best_save_path)

        scheduler.step()


def train(model, compute_loss, optimizer, optimizer_center, dataloader, device):
    model.train()
    mloss = np.array([0., 0., 0.])
    macc = 0.

    for img0, img, pid, img_path in tqdm(dataloader):
        img = img.to(device)
        pid = pid.to(device)

        optimizer.zero_grad()
        optimizer_center.zero_grad()
        score, feat = model(img)
        losses = compute_loss(score, pid, feat)
        for i in range(len(losses)):
            mloss[i] += losses[i].item()
        loss = sum(losses)
        loss.backward()
        optimizer.step()
        optimizer_center.step()

        acc = (score.max(1)[1] == pid).float().mean()
        macc += acc.item()

    mloss /= len(dataloader)
    macc /= len(dataloader)
    return mloss, macc


@torch.no_grad()
def evaluate(model, compute_loss, dataloader, device):
    model.eval()
    mloss = np.array([0., 0., 0.])
    macc = 0.

    for img0, img, pid, img_path in tqdm(dataloader):
        img = img.to(device)
        pid = pid.to(device)

        score, feat = model(img)
        losses = compute_loss(score, pid, feat)
        for i in range(len(losses)):
            mloss[i] += losses[i].item()

        acc = (score.max(1)[1] == pid).float().mean()
        macc += acc.item()

    mloss /= len(dataloader)
    macc /= len(dataloader)
    return mloss, macc


def parse_opt():
    parser = argparse.ArgumentParser()

    pre_weights = None
    parser.add_argument("--pre-weights", type=str, default=pre_weights)
    parser.add_argument("--num-classes", type=int, default=1501)
    parser.add_argument("--img-size", type=int, default=[128, 64])
    parser.add_argument("--batch-size", type=int, default=144)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--end-epoch", type=int, default=2000)
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
