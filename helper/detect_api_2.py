import torch
import torchvision.transforms as transforms

import configparser
import argparse
import os
import sys
import cv2
from pathlib import Path
import numpy as np

from recognition.CTCConverter import CTCLabelConverter
from recognition.model import Recognition_Model
from recognition.execute import *

SAVE_CWD = os.getcwd()
os.chdir(os.getcwd() + "/detection")
sys.path.append(os.getcwd())

from detection.execute import do_detect, build_detect_model

os.chdir(SAVE_CWD)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# Importing the SORT tracker
from detection.models.sort import Sort

class DetectLP:
    def __init__(self):
        self.args = None
        self.device = None
        self.converter = None
        self.detection_network = None
        self.recognition_network = None
        self.transform = None
        self.sort_tracker = Sort(max_age=300, min_hits=5, iou_threshold=0.1)
    
        self.province = ['대구서', '동대문', '미추홀', '서대문', '영등포', '인천서', '인천중',
                            '강남', '강서', '강원', '경기', '경남', '경북', '계양', '고양', '관악', '광명', '광주', '구로', '금천', '김포', '남동', 
                            '대구', '대전', '동작', '부천', '부평', '서울', '서초', '안산', '안양', '양천', '연수', '용산', '인천', '전남', '전북', 
                            '충남', '충북', '영']

        self.province_replace = ['괅', '놝', '돩', '랅', '맑', '밝', '삵', '앍', '잙', '찱',
                            '괉', '놡', '돭', '랉', '맕', '밡', '삹', '앑', '잝', '찵',
                            '괋', '놣', '돯', '뢇', '맗', '밣', '삻', '앓', '잟', '찷',
                            '괇', '놟', '돫', '뢃', '맓', '밟', '삷', '앏', '잛', '찳']

        self.chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '가', '거', '고', '구', '나', '너', '노', '누', '다', '더', '도', '두', 
                '라', '러', '로', '루', '마', '머', '모', '무', '바', '배', '버', '보', '부', '사', '서', '소', '수', '시', '아', '어', '오', 
                '우', '육', '자', '저', '조', '주', '지', '차', '카', '타', '파', '하', '허', '호', '히']

        self.chars = self.chars + self.province_replace

    def initialize(self, cfg_dir, useGPU=True):

        ################################################
        #   1. Read in parameters from config file     #
        ################################################
        if True:
            parser = argparse.ArgumentParser()

            args = parser.parse_args()

            config = configparser.RawConfigParser()
            config.read(cfg_dir)

            basic_config = config["basic_config"]

            # Weight Files
            args.detection_weight_file = basic_config["detection_weight_file"]

            args.recognition_weight_file = basic_config["recognition_weight_file"]

            # Input Data File
            args.source_vid = basic_config["source_vid"]

            # GPU Number
            args.gpu_num = basic_config["gpu_num"]
            if args.gpu_num == "" :
                print(">>> NOT Assign GPU Number")
                #sys.exit(2)

            # Detection Parameters
            args.infer_imsize_same = basic_config.getboolean('infer_imsize_same')
            if args.infer_imsize_same == "" :
                args.infer_imsize_same = False

            args.detect_save_library = basic_config.getboolean('detect_save_library')
            if args.detect_save_library == "" :
                args.detect_save_library = False

            args.data = basic_config["data"]
            if args.data == "" :
                args.data = 'detection/data/AD.yaml'

            args.half = basic_config.getboolean('half')
            if args.half == "" :
                args.half = False

            imgsz = int(basic_config["detect_imgsz"])
            args.detect_imgsz = [imgsz]
            if args.detect_imgsz == "" :
                args.detect_imgsz = [640]

            args.conf_thres = float(basic_config["conf_thres"])
            if args.conf_thres == "" :
                args.conf_thres = 0.9

            args.iou_thres = float(basic_config["iou_thres"])
            if args.iou_thres == "" :
                args.iou_thres = 0.45

            args.max_det = int(basic_config["max_det"])
            if args.max_det == "" :
                args.max_det = 1000


            # Recognition Parameters
            args.Transformation = basic_config["transformation"]
            if args.Transformation == "" :
                args.Transformation = 'TPS'

            args.FeatureExtraction = basic_config["featureExtraction"]
            if args.FeatureExtraction == "" :
                args.FeatureExtraction = 'ResNet'

            args.SequenceModeling = basic_config["sequenceModeling"]
            if args.SequenceModeling == "" :
                args.SequenceModeling = 'BiLSTM'

            args.Prediction = basic_config["prediction"]
            if args.Prediction == "" :
                args.Prediction = 'CTC'

            args.num_fiducial = int(basic_config["num_fiducial"])
            if args.num_fiducial == "" :
                args.num_fiducial = 20

            args.imgH = int(basic_config["imgH"])
            if args.imgH == "" :
                args.imgH = 64 

            args.imgW = int(basic_config["imgW"])
            if args.imgW == "" :
                args.imgW = 200

            args.output_channel = int(basic_config["output_channel"])
            if args.output_channel == "" :
                args.output_channel = 512

            args.hidden_size = int(basic_config["hidden_size"])
            if args.hidden_size == "" :
                args.hidden_size = 256

            args.input_channel = 3

            args.result_savefile = basic_config.getboolean('result_savefile')
            if args.result_savefile == "" :
                args.result_savefile = False

            args.save_detect_result = basic_config.getboolean('save_detect_result')
            if args.save_detect_result == "" :
                args.save_detect_result = False

            args.save_recog_result = basic_config.getboolean('save_recog_result')
            if args.save_recog_result == "" :
                args.save_recog_result = False

            args.hide_labels = basic_config.getboolean('hide_labels')
            if args.hide_labels == "" :
                args.hide_labels = False

            args.hide_conf = basic_config.getboolean('hide_conf')
            if args.hide_conf == "" :
                args.hide_conf = False

            args.save_conf = basic_config.getboolean('save_conf')
            if args.save_conf == "" :
                args.save_conf = False


            # Other parameters
            args.deidentified_type = basic_config["deidentified_type"]
            if args.deidentified_type == "" :
                args.deidentified_type = 2

            args.recognition_library_path = basic_config["recognition_library_path"]

        ################################################
        #        1.5 Converter for Recognition         #
        ################################################
        if True:
            converter = CTCLabelConverter(self.chars)
            args.num_class = len(converter.character)

        self.args = args
        self.converter = converter

    def set_gpu(self):

        ################################################
        #                2. Set up GPU                 #
        ################################################
        if True:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.gpu_num)
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")

    def load_networks(self):

        ################################################
        #  3. Declare Detection / Recognition Network  #
        ################################################
        if True:
            # Detection Network
            self.args.detect_weights = self.args.detection_weight_file
            detection_network, imgsz, stride, names, pt = build_detect_model(self.args, self.device)
            # Add network parameters to args
            self.args.pt = pt
            self.args.stride = stride  # YOLO 기본 stride = 32
            self.args.imgsz = imgsz  # YOLO 기본 imgsz = 640
            self.args.names = names

            print('Detection Network YOLOv5 has been successfully loaded.')

            # Recognition Network
            recognition_network = Recognition_Model(self.args, self.device)
            recognition_network.to(self.device)    

        ################################################
        #    4. Load Detection / Recognition Network   #
        ################################################
        if True:
            recognition_checkpoint = torch.load(self.args.recognition_weight_file, map_location=self.device)
            recognition_network.load_state_dict(recognition_checkpoint['network'])
            with torch.no_grad():
                detection_network.eval()
                recognition_network.eval()    

            print('Recognition Network ' + self.args.recognition_weight_file + ' has been successfully loaded.')

        self.detection_network = detection_network
        self.recognition_network = recognition_network
        self.transform = transforms.ToTensor()

    def detect(self, img_tensor, img_mat):

        img_tensor = 255 * img_tensor.permute(1, 2, 0)

        detect_preds = do_detect(self.args, self.detection_network, img_tensor, self.args.imgsz, self.args.stride, auto=True)

        return detect_preds

    def recognize(self, img_tensor, tracked_dets):
        recog_preds = []

        # 트래킹된 결과에서 유효한 바운딩 박스만 추출
        valid_bbox = []
        for track in tracked_dets:
            x1, y1, x2, y2 = map(int, track[:4])  # 바운딩 박스 좌표 추출
            if (x2 - x1) > 0 and (y2 - y1) > 0:  # 유효한 바운딩 박스 검증
                valid_bbox.append([x1, y1, x2, y2])

        if len(valid_bbox) == 0:
            return []  # 유효한 바운딩 박스가 없을 경우 빈 리스트 반환

        # numpy 배열을 PyTorch 텐서로 변환
        valid_bbox_tensor = torch.tensor(valid_bbox, device=self.device)  # GPU 사용 시 device 설정

        # 인식 수행
        recog_result = do_recognition(self.args, img_tensor, valid_bbox_tensor, self.recognition_network, self.converter, self.device)

        bbox_num = len(valid_bbox)

        cur_recog_result = []

        for bbox_idx in range(bbox_num):
            cur_bbox = valid_bbox[bbox_idx]
            x1, y1, x2, y2 = cur_bbox  # 바운딩 박스 좌표
            decode_pred = decode_province(recog_result[bbox_idx], self.province, self.province_replace)

            # 결과 저장
            append_row = [x1, y1, x2, y2, None, None, decode_pred]  # conf, cls는 None으로 처리
            cur_recog_result.append(append_row)

        recog_preds.append(cur_recog_result)

        return recog_preds


    def track(self, img_mat, detect_preds):
        # Convert detections for tracking
        dets_to_sort = np.empty((0, 6))

        # Process each detection
        for det in detect_preds:
            if det is None:
                continue
            elif det.dim() == 0:  # 0-d tensor
                print(f"Skipping 0-d tensor: {det}")
                continue
            elif det.dim() == 1:  # 1-d tensor
                if len(det) >= 6:
                    xyxy = det[:4]
                    conf = det[4]
                    cls = det[5]
                    dets_to_sort = np.vstack((dets_to_sort, np.array([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), conf.item(), cls.item()])))
                else:
                    print(f"Skipping 1-d tensor with insufficient elements: {det}")
            elif det.dim() > 1:  # 2-d tensor or higher
                for *xyxy, conf, cls in det:
                    dets_to_sort = np.vstack((dets_to_sort, np.array([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), conf.item(), cls.item()])))

        # Run the SORT tracker
        tracked_dets = self.sort_tracker.update(dets_to_sort)

        # If no detections, return empty list
        if len(tracked_dets) == 0:
            return []

        # Adjust bounding boxes to be within image bounds
        h, w, _ = img_mat.shape
        for i in range(tracked_dets.shape[0]):
            tracked_dets[i, 0] = max(0, min(tracked_dets[i, 0], w))  # x1
            tracked_dets[i, 1] = max(0, min(tracked_dets[i, 1], h))  # y1
            tracked_dets[i, 2] = max(0, min(tracked_dets[i, 2], w))  # x2
            tracked_dets[i, 3] = max(0, min(tracked_dets[i, 3], h))  # y2

        return tracked_dets

    def combine_tracking_with_recognition(self, recog_results, tracked_dets):
        # Combine tracking results with recognition
        final_results = []
        for i, track in enumerate(tracked_dets):
            track_id = track[8]

            # Append track_id to the recog_results for this detection
            decode_pred = recog_results[0][i][6]
            x1, y1, x2, y2, conf, cls = recog_results[0][i][:6]
            final_results.append([x1, y1, x2, y2, conf, cls, decode_pred, track_id])

        return final_results
    
    def file_to_torchtensor(self, imgname):
        img_mat = cv2.cvtColor(cv2.imread(imgname), cv2.COLOR_BGR2RGB)
        img_tensor = self.mat_to_torchtensor(img_mat)
        return img_mat, img_tensor

    def mat_to_torchtensor(self, img_mat):
        img_tensor = self.transform(img_mat)
        img_tensor = img_tensor.to(self.device)
        return img_tensor

if __name__ == "__main__":
    detectlp = DetectLP()
    detectlp.initialize('detect.cfg', useGPU=True)
    input_path = 'dataset/video/test1.mp4'  # replace with your input file(img/video)

    # Check if the input is a video or an image
    if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Process as video
        cap = cv2.VideoCapture(input_path)        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break           
            # Convert frame to torch tensor
            img_tensor = detectlp.mat_to_torchtensor(frame)        
            # Run tracking and recognition
            bbox = detectlp.detect(img_tensor, frame)
            recog_result = detectlp.recognize(img_tensor, bbox)
            print(recog_result)
            tracked_dets = detectlp.track(frame, bbox)
            results = detectlp.combine_tracking_with_recognition(recog_result, tracked_dets)
            print(results)

    elif input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img_mat = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
        img_tensor = detectlp.mat_to_torchtensor(img_mat)
        # Run tracking and recognition on single image
        bbox = detectlp.detect(img_tensor, img_mat)
        recog_result = detectlp.recognize(img_tensor, bbox)
        print(recog_result)
        tracked_dets = detectlp.track(img_mat, bbox)
        results = detectlp.combine_tracking_with_recognition(recog_result, tracked_dets)
        print(results)
        
    else:
        print("Unsupported file format!")