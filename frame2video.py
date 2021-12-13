import os
import sys
from pathlib import Path

import cv2

FILE = Path(__file__).absolute()
sys.path.append(os.path.join(FILE.parents[0].as_posix(), "yolov5"))
from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, CustomLoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                                  increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer,
                                  xyxy2xywh)


if __name__ == "__main__":
    source = "/media/daton/D6A88B27A88B0569/dataset/mot/MOT17/train/MOT17-04-DPM/img1"
    yolo_imgsz = [640, 640]
    stride = 32

    dir_path = "frame2videos"
    run_name = "exp"
    save_dir = increment_path(Path(dir_path) / run_name, exist_ok=False)
    save_vid = True
    if save_vid:
        save_dir.mkdir(parents=True, exist_ok=True)

    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".txt") or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)
    if webcam:
        dataset = CustomLoadStreams(source, img_size=yolo_imgsz, stride=stride, auto=True)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=yolo_imgsz, stride=stride, auto=True)
        bs = 1

    vid_path, vid_writer = [None] * bs, [None] * bs
    for path, im, im0s, vid_caps, s in dataset:
        img = im0s
        cv2.imshow(f"img", img)
        cv2.waitKey(1)

        if save_vid:
            save_path = str(save_dir / f"video")
            if vid_path != save_path:
                vid_path = save_path
                fps, w, h = 25, im0s.shape[1], im0s.shape[0]
                save_path += ".mp4"
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            vid_writer.write(img)
