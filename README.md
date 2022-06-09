# Autonomous Driving Safety Assistant
### CV 08조 주행청소년 최종프로젝트 

<br/>
<p align="center">
   <img src="https://kr.object.ncloudstorage.com/resume/boostcamp/boostcamplogo.png" />
</p>
<p align="center">
   <img src="https://kr.object.ncloudstorage.com/resume/boostcamp/boostcamplogo2.png"/>
</p>

본 과정은 NAVER Connect 재단 주관으로 인공지능과 딥러닝 Production의 End-to-End를 명확히 학습하고 실무에서 구현할 수 있도록 훈련하는 약 5개월간의 교육과정입니다. 전체 과정은 이론과정(U-stage, 5주)와 실무기반 프로젝트(P-stage, 15주)로 구성되어 있으며, 마지막 자율주제 프로젝트인 CV-08조의 자율주행간 위험예측 프로그램입니다.

<br/>

## 🚘 주행 청소년  
### 🔅 Members  

이준혁|윤서연|김 준|이재홍
:-:|:-:|:-:|:-:
![image6](https://user-images.githubusercontent.com/85532197/172898225-6b095eff-3b1d-4930-b42b-876b29214659.png) | ![image12](https://user-images.githubusercontent.com/85532197/172898232-d1405656-3b21-4f61-83e2-394d79c151e8.png) | ![image8](https://user-images.githubusercontent.com/85532197/172898238-ae6e984a-6927-4046-9430-f89bcf775cc1.png) | ![image10](https://user-images.githubusercontent.com/85532197/172898234-878b5509-66bd-4bf2-a28d-45da58635bfb.png)
[Github](https://github.com/zzundi) | [Github](https://github.com/minakusi) | [Github](https://github.com/j8n17) | [Github](https://github.com/haymrpig) 


### 🔅 Contribution  
- `이준혁` 3D Detection, Lane Projection  
- `윤서연` Forecasting, Product Serving  
- `김 준` 2D Detection, Segmentation, Product Serving
- `이재홍` 3D Detection, Tracking, Model Concatenation

<br/>

### ⚙ Development Environment

- 협업 툴 : GitHub, WandB, Notion
- 개발 환경
  - OS : Ubuntu 18.04
  - GPU : V100
  - 언어 : Python 3.7
  - dependency : Pytorch 1.7.1

# 2. Project Outline  


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


