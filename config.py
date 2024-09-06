import argparse
from random import choices

import os
import sys
from pathlib import Path

def parse_training_args(parser):
    """Add args used for training only.

    Args:
        parser: An argparse object.
    """
    # for Detection directories
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

    # Session parameters
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)

    # Source to test
    parser.add_argument('--source', type=str, default='test1.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--save_result_video', type=str2bool, default=False, help='wheter to save result video or not')
    parser.add_argument('--save_result_image', type=str2bool, default=True, help='wheter to save result images or not')

    # Saving parameters
    parser.add_argument('--save_dir', type=str, default='inference_result/', help='directory to save image results')
    parser.add_argument('--save_videoname', type=str, default='out.mp4', help='Output video name')

    parser.add_argument('--hide_labels', default=False, type=str2bool, help='hide labels')
    parser.add_argument('--hide_conf', default=False, type=str2bool, help='hide confidences')
    parser.add_argument('--half', default=False, help='use FP16 half-precision inference')

    parser.add_argument('--save_bbox', type=str2bool, default=False, help='save results to *.txt')
    parser.add_argument('--save_conf', type=str2bool, default=False, help='save confidences in --save-txt labels')
    parser.add_argument('--save_detect_img', type=str2bool, default=True, help='save detection images/videos')

    # Weights to load from
    parser.add_argument('--weight_dir', type=str, default='weights/')

    # Detection Paramters
    parser.add_argument('--detect_weights', nargs='+', type=str, default=ROOT / './weights/detection.pt', help='model path(s)')
    parser.add_argument('--data', type=str, default=ROOT / './detection/data/AD.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--detect_imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')

    parser.add_argument('--conf_thres', type=float, default=0.9, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')

    # Recognition Paramters
    parser.add_argument('--imgH', type=int, default=64)
    parser.add_argument('--imgW', type=int, default=200)
    parser.add_argument('--batch_max_length', type=int, default=9, help='Max Length of Predicted Word (7 for chinense / 9 for korean)', choices=[7, 9])
    parser.add_argument('--pad_image', type=str2bool, default=False, help='Pad when resize')

    parser.add_argument('--recognition_weight', type=str, default="recognition.pth")

    parser.add_argument('--Transformation', type=str, default='TPS', choices=['None', 'TPS'])
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', choices=['VGG, RCNN, ResNet'])
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', choices=['None', 'BiLSTM'])
    parser.add_argument('--Prediction', type=str, default='CTC', choices=['CTC', 'Attn'])

    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--img_color', type=str, default='RGB', choices=['Gray', 'RGB'])
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

def parse_args():
    """Initializes a parser and reads the command line parameters.

    Raises:
        ValueError: If the parameters are incorrect.

    Returns:
        An object containing all the parameters.
    """

    parser = argparse.ArgumentParser(description='UNet')
    parse_training_args(parser)

    return parser.parse_args()

def str2bool(v):
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

if __name__ == '__main__':
    """Testing that the arguments in fact do get parsed
    """

    args = parse_args()
    args = args.__dict__
    print("Arguments:")

    for key, value in sorted(args.items()):
        print('\t%15s:\t%s' % (key, value))
