# SNU_LP

# Environments
Pytorch >= 1.7.0

Python >= 3.7.0

```
git clone -b API --single-branch https://github.com/rjk0204/SNU_LP.git

cd SNU_LP
conda create -n ENV_NAME python=3.7
conda activate ENV_NAME
pip install -r requirements.txt
```

# Directory 설명
    |── sample_images : sample image dataset
    |── sample_video : sample video dataset
    |── detection : detection 관련 코드
    |── recognition : recognition 관련 코드
    |── weights : pretrained detection & recognition weight들 저장
    |── config.py : 입력 arugment 를 관리하는 파일
    |── detect.cfg : 입력 argument를 설정하는 파일
    └──> gulim.ttc : 한글 출력을 위한 폰트

## === 학습된 ckpt ===

아래 링크에서 미리 학습된 recognition, detection ckpt 파일을 다운 받아 weights 폴더에 배치

구글 드라이브 주소 : https://drive.google.com/drive/folders/1Tx06QJsvAUVyO73fNpC___9QPkw1ipn2?usp=sharing

## === Inference ===
```
python detect_recog_img.py
python detect_track_recog_video.py
```
#### => 실행시 {video_result/입력파일이름}/{img_result/입력파일이름}  폴더가 생성되며, 내부에 inference 결과 이미지/동영상 파일을 저장함



### [Argument (detect.cfg) 설명]


source = 입력 동영상 or 이미지 or 폴더의 경로

data = detection용 환경 setting (학습된 weight와 관련있으므로 변경하지 않는 것을 권장)

gpu_num = gpu를 사용할 수 있는 환경에서 gpu number 설정


detection_weight_file, recognition_weight_file = 각각 detection, recognition weight 파일의 경로 (변경하지 않는 것을 권장)

output_dir = inference 결과를 저장할 폴더 이름 


ex. output_dir = inference_result로 설정할 시 아래와 같이 결과 폴더가 생성됨 (주의: 같은 파일에 대해 실행시 덮어쓰기 됨)

    inference_result
        |── {입력파일 or 폴더 이름}
            |── detection : detection 결과 이미지
            |── recognition : recognition 결과 이미지
            |── label : detetction 결과 bbox label (0~1 사이로 normalized 되어 있음)     
            




### [detection 결과 저장 관련 arg]


result_savefile = 전체 결과 이미지를 저장할 지 여부

save_detect_result = detection 결과 이미지를 저장할 지 여부

hide_labels = detection 결과 이미지에서 label("LP" = License Plate)를 출력하지 않을지 여부

hide_conf = detection 결과 이미지에서 confidence 값을 출력하지 않을지 여부

save_conf = detection 결과 txt에서 confidence값을 출력하지 않을지 여부





### [recognition 결과 저장 관련 arg]


save_recog_result = recognition 결과 이미지를 저장할 지 여부

#### **주의 : labels/{파일이름}.txt 파일에는 0~1로 normalize된 center_x, center_y, w, h가 저장됨. (return 값과 다름)

 
 
 

## === Code 내부에서 return 하는 것 ===

detect_snu: 입력으로 들어온 이미지 원본과 detection predictions (bbox 좌표, confidence 값)

recognize_snu: recognition 결과 text


