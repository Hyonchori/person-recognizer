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
    source = "https://www.youtube.com/watch?v=WRp0PoxQqoQ"
    source = "https://www.youtube.com/watch?v=8B4ZKk9ow-I"
    source = "https://www.youtube.com/watch?v=WNIccic_178"
    yolo_imgsz = [640, 640]
    stride = 32

    dir_path = "youtube_videos"
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
        for i, _ in enumerate(im0s):
            print("\n---")
            print(f"{vid_caps[i].get(cv2.CAP_PROP_POS_FRAMES)} / {vid_caps[i].get(cv2.CAP_PROP_FRAME_COUNT)}")
            img = im0s[i]
            cv2.imshow(f"img{i}", im0s[i])
            cv2.waitKey(1)

            if save_vid:
                save_name = f"video_{int(vid_caps[i].get(cv2.CAP_PROP_POS_FRAMES)):05}.png"
                save_path = str(save_dir / save_name)
                print(save_path)
                cv2.imwrite(save_path, img)
