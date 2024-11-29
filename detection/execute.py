import cv2, os, glob
import torch
import numpy as np
from pathlib import Path

from utils.general import non_max_suppression, scale_coords, check_img_size, xyxy2xywh
from utils.augmentations import letterbox
from utils.plots import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from models.sort import *

def build_detect_model(args, device):
    detection_network = DetectMultiBackend(args.detect_weights, device=device, dnn=False, data=args.data, fp16=args.half)
    stride, names, pt = detection_network.stride, detection_network.names, detection_network.pt
    if len(args.detect_imgsz) ==1:
        args.detect_imgsz = [args.detect_imgsz[0], args.detect_imgsz[0]]
    imgsz = check_img_size(args.detect_imgsz, s=stride)  # check image size
    detection_network.warmup(imgsz=(1, 3, *imgsz))  # warmup

    return detection_network, imgsz, stride, names, pt


def preprocess_img(img, fp_flag, img_size, stride, auto):
    img = letterbox(img, img_size, stride=stride, auto=auto)[0]
    img = img.permute(2, 0, 1)  # HWC to CHW

    im = img.type(torch.half) if fp_flag else img.type(torch.float)  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    return im

def do_detect(args, detection_network, img, img_size, stride, auto):
    # Do Detection Inference
    im_resize = preprocess_img(img, detection_network.fp16, img_size, stride, auto)
    pred = detection_network(im_resize, augment=False, visualize=False)

    # Detection NMS
    pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, None, False, max_det=args.max_det)

    # Process predictions
    for i, det in enumerate(pred):  # per predictions
        # Rescale boxes from img_size to img size
        det[:, :4] = scale_coords(im_resize.shape[2:], det[:, :4], img.shape).round()

    return det

def save_detection_result(args, det, names, p, mode, frame, imc, save_dir):
    p = Path(p)  # to Path
    save_path = os.path.join(save_dir, p.name)  # im.jpg
    txt_path = os.path.join(save_dir, 'labels', p.stem) + ('' if mode == 'image' else f'_{frame}')  # im.txt
    gn = torch.tensor(imc.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    annotator = Annotator(imc, line_width=3, example=str(names))


    for *xyxy, conf, cls in reversed(det):
        if args.save_bbox:  # Write to file
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if args.save_conf else (cls, *xywh)  # label format
            with open(txt_path + '.txt', 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

        if args.save_detect_img:  # Add bbox to image
            c = int(cls)  # integer class
            label = None if args.hide_labels else (names[c] if args.hide_conf else f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))

    # Stream results
    im0 = annotator.result()

    # Save results (image with detections)
    if args.save_detect_img:
        if mode == 'image':
            cv2.imwrite(save_path, cv2.cvtColor(im0, cv2.COLOR_RGB2BGR))
        elif mode == 'video':
            im_save_path = save_path[:-4] + "_" + str(frame) + ".jpg"
            cv2.imwrite(im_save_path, cv2.cvtColor(im0, cv2.COLOR_RGB2BGR))

    # Print detection results
    # s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if args.save_bbox else ''


class LoadDatas:
    def __init__(self, path):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
        VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        # self.img_size = img_size
        # self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        # self.auto = auto
        self.fps = 24.0
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            self.fps = 24.0
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, f'Image Not Found {path}'

        return path, img0

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files
