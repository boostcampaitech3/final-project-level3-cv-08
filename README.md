# object-detection-level2-cv-08

# 1. Introduction  
<br/>
<p align="center">
   <img src="https://kr.object.ncloudstorage.com/resume/boostcamp/boostcamplogo.png" />
</p>
<p align="center">
   <img src="https://kr.object.ncloudstorage.com/resume/boostcamp/boostcamplogo2.png"/>
</p>

본 과정은 NAVER Connect 재단 주관으로 인공지능과 딥러닝 Production의 End-to-End를 명확히 학습하고 실무에서 구현할 수 있도록 훈련하는 약 5개월간의 교육과정입니다. 전체 과정은 이론과정(U-stage, 5주)와 실무기반 프로젝트(P-stage, 15주)로 구성되어 있으며, 두 번째 대회인 `Object detection`과제에 대한 **Level2 - 08조** 의 문제해결방법을 기록합니다.

<br/>

## 🧙‍♀️ 주행 청소년  
### 🔅 Members  

허 석|이준혁|윤서연|김 준|이재홍
:-:|:-:|:-:|:-:|:-:
 [Github](https://github.com/hursuk1) | [Github](https://github.com/zzundi) | [Github](https://github.com/minakusi) | [Github](https://github.com/j8n17) | [Github](https://github.com/haymrpig) 


### 🔅 Contribution  
- `허 석`   yolov5 model 실험 / mmdection Cascade 구조 사용 및 분석
- `이준혁` data augmentation 실험 / EfficientDet 모델 실험   
- `윤서연` EDA&pseudo labeling json 파일 생성 코드 / detectron2 라이브러리 사용하여 모델 학습  
- `김 준`   mmdetection 코드 분석 / atss, dyhead 활용 / 앙상블 
- `이재홍` analysis tool 코드 작성 / mmdetection 모델 실험 / Cross-Validation 코드 작성

<br/>

### ⚙ Development Environment

- 협업 툴 : GitHub, WandB, Notion
- 개발 환경
  - OS : Ubuntu 18.04
  - GPU : V100
  - 언어 : Python 3.7
  - dependency : Pytorch 1.7.1

# 2. Project Outline  

![image](https://user-images.githubusercontent.com/71866756/162425733-802a0a99-d368-4056-8d27-9c8e1b2c8247.png)

- Task : Object detection
- Date : 2022.03.21 - 2022.04.07 (3 weeks)
- Description : 쓰레기 사진을 입력받아서 `일반 쓰레기, 플라스틱, 종이, 유리 등`를 추측하여 `10개의 class`로 분류하고 박스의 영역을 구합니다.   
- Image Resolution : (1024 x 1024)
- Train : 4,833
- Test : 4,871

![objecteda](https://kr.object.ncloudstorage.com/resume/boostcamp/objecteda.png)


### 🏆 Final Score  
![image](https://user-images.githubusercontent.com/71866756/162425804-142bcc1c-ad37-4d13-8771-d5b9ae98e52e.png)


<br/>

# 3. Solution
### KEY POINT

- **General trash가 데이터 양에 비해 낮은 검출율을 보였다.**

  > 일반 쓰레기 범주가 너무 방대해서 생기는 문제로 보였다. 
  >
  > ( 데이터셋 자체 문제로 인한 개선 불가 )

- **클래스 간 불균형 문제**

  > Focal Loss, Over Sampling을 사용하여 개선

- **small / medium object에 대한 낮은 검출율**

  > Multiscale, base anchor size 조절, stride 조절을 통해 개선

- **높은 bias로 인한 under fitting**

  > 더 큰 backbone (Swin L)을 이용하여 개선

- **문제점 파악을 위한 Analysis Tool 사용 및 코드 작성**

<br/>

### Checklist
[More Detail](여기다가 wrap up report  링크 달기)

- [x] Test Time Augmentation
- [x] Ensemble(ATSS, Cascade R-CNN, YOLOv5x 등)
- [x] Augmentation
- [x] Multi-scale learning
- [x] Oversampling
- [x] Pseudo labeling
- [x] Stratified K-fold
- [x] Transfer learning
- [x] WandB

### Evaluation

| Method| mAP | Pseudo Labeling |
| --- | --- | --- |
|ATSS (Dyhead)| 0.6443 | O |
|Cascade R-CNN| 0.6320 |O|
|YOLOv5s|0.4492| X               |
|YOLOv5m|0.5001|X |
|YOLOv5L|0.5182|X|
|YOLOv5x|0.5984|O|
| ATSS (Dyhead) + YOLOv5x (2개 ensemble)                 | 0.6786 | X               |
|ATSS (Dyhead) + Cascade R-CNN + YOLOv5x (3개 ensemble)|0.6932|X|



# 4. Project Structure


```
├── 📂 detectron2
│   ├── 📝 train.py
│   ├── 📝 inference.py
│   └── etc
├── 📂 mmdetection
│   ├── 📂 configs
│   │   └── 📂 custom
│   ├── 📂 tools
│   │   ├── 📝 train.py
│   │   ├── 📝 test.py
│   │   └── 📝 inference.py
│   └── etc
├── 📂 yolov5
│   ├── 📝 train.py
│   ├── 📝 detect.py
│   └── etc
└── 📂 custom analysis tools
    ├── 📝 S-Kfold.py
		├── 📝 pseudo_labeling.py
    ├── 📝 analysis.ipynb
		├── 📝 ensemble.ipynb
		└── etc
```

- `detectron2`, `mmdetection`, `yolov5`에는 각각 `library file`들과 `README.md`가 존재합니다.
- `z_customs`에는 `stratified k-fold / pseudo labeling / analysis tool / ensemble` 등 자체 구현 모듈이 존재합니다.
- 각 라이브러리의 구성요소는 `README.md`에서 확인할 수 있습니다.



# 5. How to use

#### 5-1. YOLOv5

- **Train**

  ```python
  python train.py --img {img size} --batch {batch size} --epochs {epochs} --data {data yaml location} --weights {weight file loacation} --multi_scale
  ```

- **Inference**

  ```python
  python detect.py --weights {weight file location} --source {data yaml location} --img {img size} --name {save name} --half --save-txt --save-conf --augment
  ```

- **to csv**

  ```python
  python txt2csv_for_submission.py --result_path {label txt location} --save_name {save_name}
  ```

#### 5-2. detectron2

- **Train**

  ```python
  python train.py
  # cfg: .yaml 파일 변경
  # weight: cfg.MODEL.WEIGHTS 변경
  ```

- **Inference**

  ```python
  python inference.py
  # cfg: .yaml 파일 변경
  # weight: cfg.MODEL.WEIGHTS pth변경
  ```

#### 5-3. mmdetection

- **Train**

  ```python
  python train.py [config file path] --work-dir [directory path to save logs and models]
  ```

- **Inference**

  ```python
  python inference.py [config file path] [checkpoint file path] --name [submission file name]
  ```



#### 5-4. Analysis tool 

- custom analysis tools 내부에 readme 참고


