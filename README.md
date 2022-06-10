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
dlwnsgur0803@gmail.com | 이메일 | j3837301@gmail.com | 이메일

### 🔅 Contribution  
- `이준혁` 3D Detection, Lane Projection  
- `윤서연` Forecasting, Product Serving  
- `김 준` 2D Detection, Segmentation, Product Serving
- `이재홍` 3D Detection, 3D Tracking, Model Concatenation

<br/>

### ⚙ Development Environment

- 협업 툴 : GitHub, WandB, Notion
- 개발 환경
  - OS : Ubuntu 18.04
  - GPU : V100
  - 언어 : Python 3.7
  - dependency : Pytorch 1.7.1

# Project Outline  


<br/>

# Solution


# Project Structure


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




# How to use


