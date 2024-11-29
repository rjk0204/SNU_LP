import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt
import os

def save_result_img(img, recog_result, idx):

    font = ImageFont.truetype('gulim.ttc', 20)

    lp_num = len(recog_result[0])

    img_pil = Image.fromarray(img)

    for lp in recog_result[0]:

        x1, y1, x2, y2 = lp[0], lp[1], lp[2], lp[3]
        w, h = x2-x1, y2-y1
        if w*h < 800:
            recog_res = 'unknown'
        else:
            recog_res = lp[6]

        draw = ImageDraw.Draw(img_pil)
        draw.rectangle((x1,y1,x2,y2), fill=None, outline=(255,0,0))
        draw.text((x1,y1-30), recog_res, font=font, fill=(255,0,0,0))        

    out_img = np.array(img_pil)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    out_name = 'inference_result/result_' + str(idx).zfill(4) + '.png'

    cv2.imwrite(out_name, out_img)

def save_eval_results(recog_preds, video_idx, output_dir="eval_results"):
    """
    recog_preds를 파일로 저장하는 함수.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 동영상 별로 결과 저장
    output_path = os.path.join(output_dir, f"video_{video_idx}_eval_results.txt")
    
    with open(output_path, "w") as f:
        for frame_idx, results in enumerate(recog_preds):
            f.write(f"Frame {frame_idx}:\n")
            for lp in results:
                x1, y1, x2, y2, conf, cls, recog_res, track_id = lp
                f.write(f"  Bbox: ({x1}, {y1}, {x2}, {y2}), Conf: {conf}, Class: {cls}, Recognition: {recog_res}, Track ID: {track_id}\n")
    
    print(f"Results saved for video {video_idx} at {output_path}")