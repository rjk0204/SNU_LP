from detect_api import DetectLP
from dataloader import LP_Dataset
from torch.utils.data import Dataset, DataLoader
import os
import torch

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from PIL import ImageFont, ImageDraw, Image

from config import parse_args
from helpers import *

# Argument 선언
args = parse_args()
LP_Module = DetectLP()
LP_Module.initialize('detect.cfg', useGPU=True)
LP_Module.set_gpu()

GPU_NUM = 0

### Set up GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
# 현재 GPU 사용 상태 출력
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"Memory usage: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
else:
    print("No GPU available")

print(torch.cuda.is_available())  # True가 출력되어야 함
print(torch.cuda.device_count())  # GPU 개수를 확인
LP_Module.set_gpu()
print(f"LP_Module device: {LP_Module.device}")

# Dataset 선언
test_dataset = LP_Dataset(args)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    num_workers=1,
    shuffle=False,
    drop_last=False
)

# Dataset Visualize
sample_num = len(test_dataloader)
sample_idx = sorted(random.sample(range(1, len(test_dataloader)), 4))

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(40,40))
draw_imgs = []

for batch_idx, data in enumerate(test_dataloader):

    if batch_idx in sample_idx:
        _, img_mat = data

        img_mat = img_mat[0].numpy()

        draw_imgs.append(img_mat)


for i in range(2):
    ax[0][i].imshow(draw_imgs[i])
    ax[1][i].imshow(draw_imgs[i+2])

plt.xticks([])
plt.yticks([])
plt.subplots_adjust(hspace=0.01, wspace=0.1)
plt.savefig('dataset.png')

print('Image size of the samples : ', img_mat.shape)

# Netwokrks 선언
LP_Module.load_networks()

# Evaluation 진행
TEST_NUM = 2

avg_total_time = 0.0
avg_spf = 0.0

for iter_idx in range(TEST_NUM):
    recog_preds = []

    total_lp_time = 0.0

    for batch_idx, data in enumerate(tqdm(test_dataloader)):

        img_tensor, img_mat = data

        img_tensor = img_tensor[0].to(LP_Module.device)
        img_mat = img_mat[0].numpy()
        
        lp_start = time()
        bbox = LP_Module.detect(img_tensor, img_mat)
        recog_result = LP_Module.recognize(img_tensor, bbox)
        lp_end = time()

        total_lp_time += (lp_end - lp_start)

        if batch_idx in sample_idx:
            recog_preds.append(recog_result)
        
        if iter_idx == TEST_NUM-1:
            save_result_img(img_mat, recog_result, batch_idx)

    avg_lp_time = (total_lp_time / len(test_dataset))

    print('Total Time : %.3f' % (total_lp_time))
    print('Total sample num : ', len(test_dataset))
    print('Second per frame: %.3f' % (avg_lp_time))
    print('Frame per second: %.1f' % (1/avg_lp_time))
    
    avg_total_time += total_lp_time
    avg_spf += avg_lp_time

print("=========================================")
print('Average Total Time : %.3f' % (avg_total_time/TEST_NUM))
print('Total sample num : ', len(test_dataset))
print('Average Second per frame: %.3f' % (avg_spf/TEST_NUM))
print('Average Frame per second: %.1f' % (TEST_NUM/avg_spf))
print("=========================================")

# Evaluation Visualize
fig, ax = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(40,160))

font = ImageFont.truetype('gulim.ttc', 20)
result_imgs = []

for i in range(4):

    lp_num = len(recog_preds[i][0])
    img = draw_imgs[i]
    img_pil = Image.fromarray(img)

    for lp in recog_preds[i][0]:

        x1, y1, x2, y2 = lp[0], lp[1], lp[2], lp[3]
        w, h = x2-x1, y2-y1
        if w*h < 800:
            recog_res = 'unknown'
        else:
            recog_res = lp[6]
        
        draw = ImageDraw.Draw(img_pil)
        draw.rectangle((x1,y1,x2,y2), fill=None, outline=(255,0,0))
        draw.text((x1,y1-30), recog_res, font=font, fill=(255,0,0,0))
    
    img = np.array(img_pil)
    result_imgs.append(img)

for i in range(4):
    ax[i].imshow(result_imgs[i])

plt.xticks([])
plt.yticks([])
plt.subplots_adjust(hspace=0.01, wspace=0.1)    

plt.savefig('result.png')