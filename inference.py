# Inference including detection(YOLOv5), tracking, re-identification, classification

import argparse
import sys
import os
import time
import collections
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

FILE = Path(__file__).absolute()
sys.path.append(os.path.join(FILE.parents[0].as_posix(), "yolov5"))
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                                  increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer,
                                  xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync

sys.path.append(os.path.join(FILE.parents[0].as_posix(), "tracker"))
from tracker.re_identifier import Tracker

sys.path.append(os.path.join(FILE.parents[0].as_posix(), "deep_efficient_person_reid", "dertorch"))
from deep_efficient_person_reid.dertorch.nets.nn import Backbone

from deep_sort_pytorch.utils.parser import get_config as get_deepsort_cfg
from deep_sort_pytorch.deep_sort import DeepSort


def run(opt):
    # Load arguments of YOLOv5(main detector)
    yolo_weights = opt.yolo_weights
    yolo_imgsz = opt.yolo_imgsz
    yolo_conf_thr = opt.yolo_conf_thr
    yolo_iou_thr = opt.yolo_iou_thr
    yolo_max_det = opt.yolo_max_det
    yolo_target_clss = opt.yolo_target_clss
    yolo_face_mosaic = opt.yolo_face_mosaic
    yolo_save_crop = opt.yolo_save_crop
    yolo_save_interval = opt.yolo_save_interval

    # Load arguments of EfficientNet2(feature extractor)
    effnet_weights = opt.effnet_weights
    effnet_imgsz = opt.effnet_imgsz

    # Load arguments of tracker(custom re-identifier)
    tracker_queries = opt.tracker_queries

    # Load arguments of DeepSORT(main tracker)
    deepsort_cfg = get_deepsort_cfg()
    deepsort_cfg.merge_from_file(opt.deepsort_cfg)
    deepsort_cfg["DEEPSORT"]["REID_CKPT"] = opt.deepsort_weights

    # Load general arguments
    source = opt.source
    device = opt.device
    dir_path = opt.dir_path
    run_name = opt.run_name
    is_video_frames = opt.is_video_frames
    show_cls = opt.show_cls
    save_vid = opt.save_vid
    hide_labels = opt.hide_labels
    hide_conf = opt.hide_conf
    use_model = opt.use_model
    show_model = {key: use_model[key] & opt.show_model[key] for key in use_model}

    # Initialize setting
    device = select_device(device)
    save_dir = increment_path(Path(dir_path) / run_name, exist_ok=False)
    if save_vid:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLOv5 model
    print("############### Load YOLOv5 for main detector ################")
    yolo_model = DetectMultiBackend(yolo_weights, device=device, dnn=False)
    stride, names, pt, jit, onnx = yolo_model.stride, yolo_model.names, yolo_model.pt, yolo_model.jit, yolo_model.onnx
    yolo_imgsz = check_img_size(yolo_imgsz, s=stride)

    # Load EfficientNet2 model
    print("\n############### Load EfficientNetv2 for feature extractor ################")
    effnet_model = Backbone(num_classes=255, model_name='efficientnet_v2').to(device)
    effnet_model.load_param(effnet_weights)
    effnet_model.eval()
    effnet_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Tracker using feature extractor
    print("\n############### Load custom tracker for re-identification ################")
    tracker = Tracker(query_root=tracker_queries,
                      feature_extractor=effnet_model,
                      transform=effnet_transform,
                      device=device,
                      img_size=effnet_imgsz)

    tracker.plot_pca()
    #sys.exit()

    # DataLoader
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".txt") or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)
    if webcam:
        cudnn.banchmark = True
        dataset = LoadStreams(source, img_size=yolo_imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=yolo_imgsz, stride=stride, auto=pt and not jit)
        bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Load DeepSORT model
    deepsort_model_list = [DeepSort(deepsort_cfg.DEEPSORT.REID_CKPT,
                                    max_dist=deepsort_cfg.DEEPSORT.MAX_DIST,
                                    min_confidence=deepsort_cfg.DEEPSORT.MIN_CONFIDENCE,
                                    max_iou_distance=deepsort_cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                    max_age=deepsort_cfg.DEEPSORT.MAX_AGE,
                                    n_init=deepsort_cfg.DEEPSORT.N_INIT,
                                    nn_budget=deepsort_cfg.DEEPSORT.NN_BUDGET,
                                    use_cuda=True) for _ in range(bs)]

    # Initial inference
    if pt and device.type != "cpu":
        yolo_model(torch.zeros(1, 3, *yolo_imgsz).to(device).type_as(next(yolo_model.model.parameters())))
        effnet_model(torch.zeros(1, 3, *effnet_imgsz).to(device).type_as(next(effnet_model.parameters())))

    # Run inference
    yolo_save_count = 0
    for path, im, im0s, vid_cap, s in dataset:
        print("\n---")

        # Preprocess image for YOLOv5
        t1 = time_sync()
        im = torch.from_numpy(im).to(device).float() / 255.
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # YOLOv5 inference
        if use_model["yolov5"]:
            yolo_pred = yolo_model(im, augment=False, visualize=False)

            # NMS
            yolo_pred = non_max_suppression(yolo_pred, yolo_conf_thr, yolo_iou_thr, yolo_target_clss,
                                            agnostic=False, max_det=yolo_max_det)
        else:
            yolo_pred = []

        for i, det in enumerate(yolo_pred):
            if webcam:
                p, im0, imv, frame = path[i], im0s[i].copy(), im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, imv, frame = path, im0s.copy(), im0s.copy(), getattr(dataset, "frame", 0)
            p = Path(p)
            save_path = str(save_dir / "video") if is_video_frames else str(save_dir / p.name)
            annotator = Annotator(imv, line_width=2, example=str(names))

            if len(det) > 0:
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], imv.shape).round()

                effnet_input = []
                for box in det:
                    if use_model["tracker"]:
                        if box[-1] == 0:
                            tmp_ref = im0[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                            tmp = cv2.resize(tmp_ref, dsize=[effnet_imgsz[0], effnet_imgsz[1]])
                            tmp = effnet_transform(tmp)[None]
                            effnet_input.append(tmp)

                    if yolo_face_mosaic:
                        if box[-1] == 1 or box[-1] == 2:  # cls 1: unsure head / 2: head(face)
                            head_ref = imv[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                            if 0 in head_ref.shape:
                                continue
                            head = cv2.resize(head_ref, dsize=(10, 10))
                            head = cv2.resize(head, head_ref.shape[:2][::-1], interpolation=cv2.INTER_AREA)
                            imv[int(box[1]): int(box[3]), int(box[0]): int(box[2])] = head

                    if yolo_save_crop:
                        if yolo_save_count % yolo_save_interval == 0 and box[-1] == 0:  # cls 0: person(full body)
                            save_one_box(box[:4], im0, file=save_dir / "crops" / names[int(box[-1])] / f"{p.stem}.jpg",
                                         BGR=True)
                            yolo_save_count = 0

                if use_model["deepsort"]:
                    clss = det[:, -1]
                    person_idx = clss == 0
                    xywhs = xyxy2xywh(det[:, :4])[person_idx]
                    confs = det[:, 4][person_idx]
                    clss = clss[person_idx]
                    deepsort_pred = deepsort_model_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                box_show = show_model["yolov5"] or show_model["deepsort"]
                if show_model["yolov5"] and not show_model["deepsort"]:
                    box_iter = det
                elif show_model["deepsort"]:
                    box_iter = deepsort_pred
                else:
                    box_iter = det

                for box in box_iter:
                    xyxy = box[:4]
                    conf = box[4]
                    cls = int(box[5])

                    if box_show and cls in show_cls:
                        id = cls if not use_model["deepsort"] else int(box[-1])
                        label = None if hide_labels else (names[cls] if hide_conf else f'{id} {names[cls]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(id, True))
            else:
                if use_model["deepsort"]:
                    deepsort_model_list[i].increment_ages()

            tf = time_sync()
            print(f"{s}Done. ({tf - t1:.3f}s)")

            # Visualize results
            if any(show_model.values()):
                cv2.imshow(f"img{i}", imv)
                cv2.waitKey(66)

            # Save results
            if save_vid:
                if dataset.mode == "image" and not is_video_frames:
                    cv2.imwrite(save_path, imv)
                elif dataset.mode == "image" and is_video_frames:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        fps, w, h = 30, imv.shape[1], imv.shape[0]
                        save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(imv)
                else:
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, imv.shape[1], imv.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(imv)
        yolo_save_count += 1


def parse_opt():
    parser = argparse.ArgumentParser()

    # Arguments for YOLOv5(main person detector), cls 0: person / 1: unsure head / 2: head(face)
    yolo_weights = f"{FILE.parents[0]}/weights/yolov5/yolov5l_crowdhuman_v7.pt"
    #yolo_weights = f"{FILE.parents[0]}/weights/yolov5/firedetector_v1.pt"
    parser.add_argument("--yolo-weights", nargs="+", type=str, default=yolo_weights)
    parser.add_argument("--yolo-imgsz", "--yolo-img-size",  type=int, default=[640])
    parser.add_argument("--yolo-conf-thr", type=float, default=0.5)
    parser.add_argument("--yolo-iou-thr", type=float, default=0.7)
    parser.add_argument("--yolo-max-det", type=int, default=300)
    parser.add_argument("--yolo-target-clss", nargs="+", default=None)  # [0, 1, 2, ...]
    parser.add_argument("--yolo-face-mosaic", default=False)  # apply mosaic to unsure_head and head
    parser.add_argument("--yolo-save-crop", default=False)
    parser.add_argument("--yolo-save-interval", type=int, default=5)

    # Arguments for EfficientNet2(feature extractor for re-identification)
    effnet_weights = f"{FILE.parents[0]}/weights/efficientnet2/efficientnet_v2_model_300.pth"
    parser.add_argument("--effnet-weights", type=str, default=effnet_weights)
    parser.add_argument("--effnet-imgsz", "--effnet-img-size", type=int, default=[128, 256])

    # Arguments for Tracker(custom re-identifier)
    tracker_queries = f"{FILE.parents[0]}/queries"
    parser.add_argument("--tracker-queries", type=str, default=tracker_queries)

    # Arguments for DeepSORT(main tracker)
    parser.add_argument("--deepsort-cfg", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument("--deepsort-weights", type=str, default="weights/deep_sort/deep/checkpoint/ckpt.t7")

    # General arguments
    source = "rtsp://datonai:datonai@172.30.1.49:554/stream1"
    source = "/media/daton/Data/datasets/MOT17/train/MOT17-04-FRCNN/img1"
    #source = "https://www.youtube.com/watch?v=668J-hyfJ0E"
    #source = "https://www.youtube.com/watch?v=WRp0PoxQqoQ"
    #source = "/media/daton/D6A88B27A88B0569/dataset/화재_발생_예측_영상/Validation/[원천]화재씬2/S3-N0819MF06491.jpg"
    #source = "https://www.youtube.com/watch?v=kR5h18Jdcyc"
    #source = "0"
    #source = "/home/daton/Downloads/daton_office_02-people_counting.mp4"
    parser.add_argument("--source", type=str, default=source)
    parser.add_argument("--device", default="")
    parser.add_argument("--dir-path", default="runs/inference")
    parser.add_argument("--run-name", default="exp")
    parser.add_argument("--is-video-frames", type=bool, default=True)  # use when process images from video
    parser.add_argument("--show-cls", type=int, default=[0])  # [0, 1, 2, ...]
    parser.add_argument("--save-vid", type=bool, default=True)
    parser.add_argument("--hide-labels", type=bool, default=False)
    parser.add_argument("--hide-conf", type=bool, default=False)
    parser.add_argument("--use-model", type=dict,
                        default={"yolov5": True,
                                 "deepsort": False,
                                 "tracker": False})
    parser.add_argument("--show-model", type=dict,
                        default={"yolov5": True,
                                 "deepsort": False,
                                 "tracker": False})

    opt = parser.parse_args()
    opt.yolo_imgsz *= 2 if len(opt.yolo_imgsz) == 1 else 1
    return opt


def main(opt):
    print(colorstr('Inference: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    run(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)