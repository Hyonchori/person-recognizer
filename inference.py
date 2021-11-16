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

FILE = Path(__file__).absolute()
sys.path.append(os.path.join(FILE.parents[0].as_posix(), "tracker"))
from tracker.person import Person



def run(opt):
    # Load arguments of YOLOv5(main detector)
    yolo_weights = opt.yolo_weights
    yolo_imgsz = opt.yolo_imgsz
    yolo_conf_thr = opt.yolo_conf_thr
    yolo_iou_thr = opt.yolo_iou_thr
    yolo_max_det = opt.yolo_max_det
    yolo_target_clss = opt.yolo_target_clss
    yolo_face_mosaic = opt.yolo_face_mosaic

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
    show_model = opt.show_model

    # Initialize setting
    device = select_device(device)
    save_dir = increment_path(Path(dir_path) / run_name, exist_ok=False)
    if save_vid:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLOv5 model
    model = DetectMultiBackend(yolo_weights, device=device, dnn=False)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
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
        dataset = LoadImages(source, img_size=yolo_imgsz, stride=stride, auto=pt and not jit)
        bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Initial inference
    if pt and device.type != "cpu":
        model(torch.zeros(1, 3, *yolo_imgsz).to(device).type_as(next(model.model.parameters())))

    # Run inference
    dt, seen = [0., 0., 0.], 0
    for path, im, im0s, vid_cap, s in dataset:
        print("\n---")

        # Preprocess image for YOLOv5
        t1 = time_sync()
        im = torch.from_numpy(im).to(device).float() / 255.
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # YOLOv5 inference
        if use_model["yolov5"]:
            yolo_pred = model(im, augment=False, visualize=False)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            yolo_pred = non_max_suppression(yolo_pred, yolo_conf_thr, yolo_iou_thr, yolo_target_clss,
                                            agnostic=False, max_det=yolo_max_det)
            dt[2] += time_sync() - t3
        else:
            yolo_pred = []

        for i, det in enumerate(yolo_pred):
            seen += 1
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
                    if use_model["tracker"]:
                        if box[-1] == 0:
                            pass
                            tmp_person = Person(im0, box[:4])

                    if yolo_face_mosaic:
                        if box[-1] == 1 or box[-1] == 2:  # cls 1: unsure head / 2: head(face)
                            head_ref = imv[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                            if 0 in head_ref.shape:
                                continue
                            head = cv2.resize(head_ref, dsize=(10, 10))
                            head = cv2.resize(head, head_ref.shape[:2][::-1], interpolation=cv2.INTER_AREA)
                            imv[int(box[1]): int(box[3]), int(box[0]): int(box[2])] = head



                box_show = show_model["yolov5"]
                if show_model["yolov5"]:
                    box_iter = det
                else:
                    box_iter = det

                for box in box_iter:
                    xyxy = box[:4]
                    conf = box[4]
                    cls = int(box[-1])



                    if box_show and cls in show_cls:
                        id = cls
                        label = None if hide_labels else (names[cls] if hide_conf else f'{names[cls]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(id, True))

            LOGGER.info(f"{s}Done. ({t3 - t2:.3f}s)")

            # Visualize results
            if any(show_model.values()):
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




def parse_opt():
    parser = argparse.ArgumentParser()

    # Arguments for YOLOv5(main person detector), cls 0: person / 1: unsure head / 2: head(face)
    yolo_weights = "weights/yolov5/yolov5l_crowdhuman_v4.pt"
    parser.add_argument("--yolo-weights", nargs="+", type=str, default=yolo_weights)
    parser.add_argument("--yolo-imgsz", "--yolo-img-size",  type=int, default=[640])
    parser.add_argument("--yolo-conf-thr", type=float, default=0.5)
    parser.add_argument("--yolo-iou-thr", type=float, default=0.6)
    parser.add_argument("--yolo-max-det", type=int, default=300)
    parser.add_argument("--yolo-target-clss", nargs="+", default=None)  # [0, 1, 2, ...]
    parser.add_argument("--yolo-face-mosaic", default=True)  # apply mosaic to unsure_head and head

    # General arguments
    source = "rtsp://datonai:datonai@172.30.1.49:554/stream1"
    source = "/media/daton/D6A88B27A88B0569/dataset/mot/MOT17/train/MOT17-02-DPM/img1"
    #source = "0"
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
                                 "tracker": True})
    parser.add_argument("--show-model", type=dict,
                        default={"yolov5": True,
                                 "tracker": True})

    opt = parser.parse_args()
    opt.yolo_imgsz *= 2 if len(opt.yolo_imgsz) == 1 else 1
    return opt


def main(opt):
    print(colorstr('Inference: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    run(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)