import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from helper.detect_api_2 import DetectLP
from config import parse_args
import time

# DetectLP 초기화
args = parse_args()
LP_Module = DetectLP()
LP_Module.initialize('detect.cfg', useGPU=True)
LP_Module.set_gpu()
LP_Module.load_networks()

# 폰트 경로 설정
font_path = 'gulim.ttc'  # 글꼴 경로

def process_and_save_video(input_video_path, text_color, LP_Module):
    # 출력 디렉토리와 파일 경로 생성
    output_dir = "./video_result"
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, "output.mp4")

    # 비디오 캡처 초기화
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {input_video_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print(f"Processing video: {input_video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    font = ImageFont.truetype(font_path, 20)

    for _ in tqdm(range(frame_count), desc="Processing Frames"):
        ret, frame = cap.read()
        if not ret:
            break

        img_mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = LP_Module.mat_to_torchtensor(img_mat)

        # Detection -> Tracking -> Recognition
        bbox = LP_Module.detect(img_tensor, img_mat)  
        tracked_dets = LP_Module.track(img_mat, bbox)  
        recog_result = LP_Module.recognize(img_tensor, tracked_dets)  
        final_results = LP_Module.combine_tracking_with_recognition(recog_result, tracked_dets)


        # Visulize
        img_pil = Image.fromarray(img_mat)
        draw = ImageDraw.Draw(img_pil)
        for result in final_results:
            x1, y1, x2, y2 = map(int, result[:4])
            conf, cls, recog_text, track_id = result[4], result[5], result[6], result[7]
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

            text = f"ID: {int(track_id)}, Recog: {recog_text}"
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
            for dx, dy in offsets:
                draw.text((x1 + dx, y1 - 10 + dy), text, font=font, fill=text_color)

        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        out.write(img_cv)

    cap.release()
    out.release()
    print(f"Saved processed video to: {output_video_path}")
    return output_video_path

if __name__ == "__main__":
    input_video_path = "./sample_video/test1.mp4" 
    text_color = "#FFFFFF"  # (흰색: #FFFFFF, 검은색: #000000)

    process_and_save_video(input_video_path, text_color, LP_Module)