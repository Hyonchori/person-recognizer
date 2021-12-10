# Inference including detection(YOLOv5), tracking, re-identification, classification

import argparse
import sys
import os
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


def run(opt):
    # Load arguments of YOLOv5(main detector)
    yolo_weights = opt.yolo_weights
    yolo_imgsz = opt.yolo_imgsz
    yolo_conf_thr = opt.yolo_conf_thr
    yolo_iou_thr = opt.yolo_iou_thr
    yolo_max_det = opt.yolo_max_det
    yolo_target_clss = opt.yolo_target_clss
    yolo_save_crop = opt.yolo_save_crop
    yolo_save_interval = opt.yolo_save_interval
    yolo_save_txt = opt.yolo_save_txt

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
    show_vid = opt.show_vid

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
        import glob
        dataset = LoadImages(source, img_size=yolo_imgsz, stride=stride, auto=pt and not jit)
        bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Initial inference
    if pt and device.type != "cpu":
        yolo_model(torch.zeros(1, 3, *yolo_imgsz).to(device).type_as(next(yolo_model.model.parameters())))

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
        yolo_pred = yolo_model(im, augment=False, visualize=False)

        # NMS
        yolo_pred = non_max_suppression(yolo_pred, yolo_conf_thr, yolo_iou_thr, yolo_target_clss,
                                        agnostic=False, max_det=yolo_max_det)

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

                for box in det:
                    if yolo_save_crop:
                        if yolo_save_count % yolo_save_interval == 0 and box[-1] == 0:  # cls 0: person(full body)
                            save_one_box(box[:4], im0, file=save_dir / "crops" / names[int(box[-1])] / f"{p.stem}.jpg",
                                         BGR=True)
                            yolo_save_count = 0

                for box in det:
                    xyxy = box[:4]
                    conf = box[4]
                    cls = int(box[5])

                    if show_vid and cls in show_cls:
                        id = cls
                        label = None if hide_labels else (names[cls] if hide_conf else f'{id} {names[cls]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(id, True))

            tf = time_sync()
            print(f"{s}Done. ({tf - t1:.3f}s)")

            # Visualize results
            if show_vid:
                cv2.imshow(f"img{i}", imv)
                cv2.waitKey(1)

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

    # Arguments for YOLOv5(main person detector)
    # "black smoke", "gray smoke", "white smoke", "fire", "cloud", "fog", "light", "sun light", "swing1", "swing2"
    yolo_weights = f"{FILE.parents[0]}/weights/yolov5/fire_v1.pt"
    parser.add_argument("--yolo-weights", nargs="+", type=str, default=yolo_weights)
    parser.add_argument("--yolo-imgsz", "--yolo-img-size",  type=int, default=[640])
    parser.add_argument("--yolo-conf-thr", type=float, default=0.5)
    parser.add_argument("--yolo-iou-thr", type=float, default=0.7)
    parser.add_argument("--yolo-max-det", type=int, default=300)
    parser.add_argument("--yolo-target-clss", nargs="+", default=None)  # [0, 1, 2, ...]
    parser.add_argument("--yolo-save-crop", default=False)
    parser.add_argument("--yolo-save-interval", type=int, default=5)
    parser.add_argument("--yolo-save-txt", type=bool, default=True)

    # General arguments
    source = "rtsp://datonai:datonai@172.30.1.49:554/stream1"
    source = "/media/daton/D6A88B27A88B0569/dataset/fire detection/total"
    #source = "0"
    parser.add_argument("--source", type=str, default=source)
    parser.add_argument("--device", default="")
    parser.add_argument("--dir-path", default="runs/fire_detect")
    parser.add_argument("--run-name", default="exp")
    parser.add_argument("--is-video-frames", type=bool, default=False)  # use when process images from video
    parser.add_argument("--show-vid", type=bool, default=False)
    parser.add_argument("--show-cls", type=int, default=[x for x in range(4)])  # [0, 1, 2, ...]
    parser.add_argument("--save-vid", type=bool, default=False)
    parser.add_argument("--hide-labels", type=bool, default=False)
    parser.add_argument("--hide-conf", type=bool, default=False)

    opt = parser.parse_args()
    opt.yolo_imgsz *= 2 if len(opt.yolo_imgsz) == 1 else 1
    return opt


def main(opt):
    print(colorstr('Inference: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    run(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)