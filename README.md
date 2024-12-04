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
#### => 실행시 {video_result/입력파일이름}/{img_result/입력파일이름}  폴더가 생성되며, 
#### 내부에 inference 결과 이미지/동영상 파일을 저장함



### [Argument (detect.cfg) 설명]


data = detection용 환경 setting (학습된 weight와 관련있으므로 변경하지 않는 것을 권장)

gpu_num = gpu를 사용할 수 있는 환경에서 gpu number 설정


detection_weight_file, recognition_weight_file = 각각 detection, recognition weight 파일의 경로 (변경하지 않는 것을 권장)
source_img = image sample 경로, inference 하고자 하는 image 경로로 변경
source_vid = video sample 경로, inference 하고자 하는 video 경로로 변경

## === Code 내부에서 return 하는 것 ===

output_path/output_video_path: 저장하는 결과 이미지/동영상의 경로
