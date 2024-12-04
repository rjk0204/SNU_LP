import os
from PIL import Image, ImageDraw, ImageFont
import cv2
from helper.detect_api import DetectLP
import argparse
import configparser

# DetectLP 모듈 초기화
def initialize_lp_module(cfg_dir, useGPU=True):
    LP_Module = DetectLP()
    LP_Module.initialize(cfg_dir, useGPU)
    LP_Module.set_gpu()
    LP_Module.load_networks()
    return LP_Module

def process_and_save_image(image_path, text_color, LP_Module):
    # 결과 저장 디렉토리 생성
    output_dir = "./img_result"
    os.makedirs(output_dir, exist_ok=True)

    # 입력 이미지 파일 이름에서 확장자 제거
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{filename}_processed.png")

    # 이미지를 OpenCV 형식으로 변환
    img_mat = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img_tensor = LP_Module.mat_to_torchtensor(img_mat)

    # Detection 및 Recognition 수행
    bbox = LP_Module.detect(img_tensor, img_mat)
    recog_result = LP_Module.recognize(img_tensor, bbox)

    # 시각화
    img_pil = Image.fromarray(img_mat)
    draw = ImageDraw.Draw(img_pil)
    font_path = 'gulim.ttc'  # 폰트 경로
    font = ImageFont.truetype(font_path, 20)  # 글씨 크기 설정

    # bbox와 recog_result를 매칭하여 시각화
    for i, b in enumerate(bbox):
        if len(b) >= 4:
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

            # recog_result에서 매칭된 텍스트 가져오기
            recog_text = recog_result[0][i][-1] if recog_result and len(recog_result[0]) > i else "Unknown"

            # 텍스트 굵기를 위해 여러 번 겹쳐 그리기
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
            for dx, dy in offsets:
                draw.text((x1 + dx, y1 - 10 + dy), recog_text, font=font, fill=text_color)

    # 처리된 이미지 저장
    img_pil.save(output_path)
    print(f"Processed image saved at: {output_path}")
    return output_path

# 직접 실행 부분
if __name__ == "__main__":
    # Config 파일에서 경로를 읽어오기
    config = configparser.ConfigParser()
    config.read('detect.cfg')  # detect.cfg 파일을 읽음

    # config 파일에서 'basic_config' 섹션에 있는 'source_img' 경로를 가져옴
    input_image_path = config['basic_config']['source_img']
    text_color = "white"  # 텍스트 색상 (white 또는 black)

    LP_Module = initialize_lp_module('detect.cfg', useGPU=True)

    # 이미지 처리 및 저장
    process_and_save_image(input_image_path, text_color, LP_Module)
