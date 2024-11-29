from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import sys

class LP_Dataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.video_paths = []  # 비디오 경로 리스트

        # args.data_dir에서 비디오 파일 경로를 읽어옴
        for file in os.listdir(os.path.join(args.data_dir)):
            if file.endswith(".mp4"):  # mp4 파일만 가져옴
                self.video_paths.append(os.path.join(args.data_dir, file))

        # 각 비디오 파일에서 프레임 수를 저장할 리스트
        self.frames_list = []
        self.total_frames = 0  # 전체 프레임 수

        # 각 비디오 파일의 프레임 수를 계산
        for video_path in self.video_paths:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frames_list.append(frame_count)
            self.total_frames += frame_count
            cap.release()

        # ToTensor 변환 설정
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.total_frames  # 전체 비디오의 총 프레임 수 반환

    def __getitem__(self, index):
        video_idx, frame_idx = self.get_video_and_frame_index(index)
        return self.get_frame_from_video(self.video_paths[video_idx], frame_idx)

    def get_video_and_frame_index(self, index):
        cumulative_frames = 0
        for video_idx, frames in enumerate(self.frames_list):
            if index < cumulative_frames + frames:
                frame_idx = index - cumulative_frames
                return video_idx, frame_idx
            cumulative_frames += frames
        raise IndexError("인덱스가 총 프레임 수를 초과했습니다.")

    def get_frame_from_video(self, video_path, frame_idx):
        # 특정 비디오에서 해당 프레임을 불러오는 함수
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img_mat = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"프레임 {frame_idx}를 {video_path}에서 불러올 수 없습니다.")
        
        img_mat = cv2.cvtColor(img_mat, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환
        img_tensor = self.transform(img_mat)  # Tensor로 변환
        return img_tensor, img_mat

    def get_video_info(self):
        # 각 비디오의 이름과 프레임 수를 반환
        video_info = []
        for i, video_path in enumerate(self.video_paths):
            video_name = os.path.basename(video_path)
            frame_count = self.frames_list[i]
            video_info.append(f"{video_name}: {frame_count}")
        return video_info
    
    def get_video_frames(self, video_idx):
        # 해당 비디오의 모든 프레임을 순차적으로 반환하는 함수
        video_path = self.video_paths[video_idx]
        cap = cv2.VideoCapture(video_path)
        video_frames = []
        for frame_idx in range(self.frames_list[video_idx]):
            ret, img_mat = cap.read()
            if not ret:
                continue
            img_mat = cv2.cvtColor(img_mat, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환
            img_tensor = self.transform(img_mat)  # Tensor 변환
            video_frames.append((img_tensor, img_mat))
        cap.release()
        return video_frames
