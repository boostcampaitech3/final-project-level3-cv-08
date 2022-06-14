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
dlwnsgur0803@gmail.com | seoyeon737@gmail.com | j3837301@gmail.com | nevermail_@naver.com

### 🔅 Contribution  
- `이준혁` - 3D Detection, Lane Projection  
- `윤서연` - Forecasting, Product Serving  
- `김 준` - 2D Detection, Segmentation, Product Serving
- `이재홍` - 3D Detection, 3D Tracking, Model Concatenation

<br/>

### ⚙ Development Environment

- 협업 툴 : GitHub, WandB, Notion
	- 협업을 위해 노션 칸반보드 적극활용 (https://pinto-throne-474.notion.site/186f4c294fd44320a8270aa2c7ddd96c)
- 개발 환경
  - OS : Ubuntu 18.04
  - GPU : V100
  - 언어 : Python 3.7
  - dependency : Pytorch 1.7.1

# Project Outline  
## 프로젝트 주제

카메라를 이용한 자율주행을 상품화한 테슬라의 등장으로 자율주행 시장은 더욱 활성화가 되어 가고 있습니다.

하지만, 카메라가 담을 수 있는 정보에는 한계가 존재하여, 자율주행 중인 테슬라 차량이 큰 트레일러 차량을 들이 박는 사고가 종종 일어났습니다.
객체의 위치정보를 보다 정확하게 알 수 있는 라이다를 이용하여 카메라 정보와 융합하여
사용한다면, 보다 안전한 자율주행이 가능할 것입니다.

이 프로젝트에서는 라이다 데이터와 카메라 데이터를 함께 이용하여 객체 추정, 이동 예측을 통하여 주행 차량 앞으로 보행자나 차량이 끼어드는 것을 경고하여 보다 안전한 자율주행을 위한 서비스를 제작하였습니다.

## 기대 효과
<img src='https://user-images.githubusercontent.com/85532197/173238024-e62e2925-226c-4576-9926-602ba74109dd.png' width=50%><img src='https://user-images.githubusercontent.com/85532197/173237976-7436d213-2bc7-4a6e-8f27-aa5470e70200.png' width=50%>

사람의 눈으로는 모든 상황에 대해서 한눈에 확인할 수 없습니다. 만약 딥러닝을 통해 사람을 보조할 수 있다면 사고의 위험성은 크게 줄어들 것입니다. 

라이다, 카메라 데이터를 통해서 detection, tracking, forecasting을 통해 위험한 상황을 예측하고 사용자에게 경고해줌으로써 사고를 줄일 수 있습니다. 

또한 더 나아가서 비단 경고 뿐 아니라 실제 차량 제어를 통해 자율주행 level 4까지 나아갈 수 있을 것입니다.

## Dataset
- **KITTI Dataset**
    - **LIDAR**
        
        3D point cloud 데이터
        
        > 3D detection을 위한 세개의 클래스 (`Pedestrian`, `Car`, `Cyclist`)
        > 
    - **CAMERA**
        
        2D image 데이터
        
        > 2D detection을 위한 데이터
        > 
    - **GPS / IMU**
        
        GPS와 IMU 정보
        
        > ego 차량의 이동량을 파악하기 위한 GPS와 IMU 데이터
- **BDD Dataset**
    - **CAMERA**
        
        2D image 데이터
        
        > 2D 차선 및 주행가능영역 Segmentation 데이터
        >
## Model Structure
![Image](https://user-images.githubusercontent.com/85532197/173237401-fde65883-2410-4883-beb2-ee03e0df59f8.png)
- 2D - YOLOP (Detection - YOLOv5, Segmentation - Seg head 2개 추가)
- 3D - Pointpillars
- Tracking - Sort
- Forecasting - PECNet

## Demo Page Structure
![Image](https://user-images.githubusercontent.com/85532197/173237405-07fcaf37-7356-4d1d-b636-6ee4851f45ad.png)
- Frontend - Streamlit
- ackend - FastAPI
- Storage - Google Cloud Storage

# Demo
![FUSION_result_A](https://user-images.githubusercontent.com/85532197/173238845-7198f4c7-1d8b-4f21-8906-89ce3f896396.gif)


<br/>

# Project Structure


```
├── 📂 serving
│   ├── 📂 app
│   │   ├── 📝 __main__.py
│   │   ├── 📝 frontend.py
│   │   ├── 📝 main.py
│   │   └── 📝 model.py
│   ├── 📂 data
│   │   └── 📝 pth files
│   ├── 📂 deep_sort
│   |	├── 📂 training
│   |	└── 📂 testing
│   ├── 📂 utils
|   └── 📂 lib_bdd
└── 📂 YOLOP
|   ├── 📂 lib
|   ├── 📂 tools
|   |   ├── 📝 demo.py
|   |   ├── 📝 train.py
|   |   └── 📝 test.py
|   └── etc
└── 📂 mmdetection3d
    ├── 📂 tools
    |   ├── 📂 custom_tools
    |   |   └── 📝 make_videos.ipynb
    |   ├── 📂 utils
    |   |   └── 📝 utils.py
    |   ├── 📝 train.py
    |   └── 📝 test.py
    └── etc
```
- 프로젝트 구조
    - Serving - Streamlit-FastAPI로 생성한 데모 사이트에 대한 코드 존재
    - YOLOP - 2D Detection 및 Semantic Segmentation에 대한 코드 존재
    - mmdetection3d - 2D Detection & Segmentation, 3D Detection, Tracking, Forecasting을 합친 코드 존재
