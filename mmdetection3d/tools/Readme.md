# 1. Tracking Kitti Dataset 구조 만들기

### 1-1. 폴더 구조 만들기

```
📂 kitti
├── 📂 ImageSets
│   └── 📝 test.txt
├── 📂 oxts
│   	└── 📂 data
│			├── 📝 000000.txt
│   		├── 📝 000001.txt
│   				...
│   		└──	📝 000xxx.txt
└──── 📂 testing
				├── 📂 calib
				├── 📂 velodyne
				└── 📂 image_2
```

- **ImageSets의 test.txt**

  이미지 / bin파일의 index가 저장되어 있다. 

  ```
  000000
  000001
  000002
  000003
  000004
  000005
  ...
  ```

- **oxts**

  GPS / IMU 데이터가 저장되어 있다. 

- **testing**

  - calib

    tracking의 경우 calibration 파일은 하나이므로, xxx.txt형태로 하나의 파일이 저장되어 있다. 

  - image_2

    모든 이미지들이 저장되어 있다. (파일이름 6자리 숫자)

  - velodyne

    모든 bin파일 저장되어 있다. (파일이름 6자리 숫자)

### 1-2. mmdetection3d에 필요한 pickle파일 생성하기

- ..../mmdetection3d/tools/create_data.py 실행

  ```python
  python create_data.py kitti --root-path "1-1의 폴더 경로" --tracking True --out-dir "1-1의 폴더 경로" --extra-tag kitti
  ```

# 2. inference하기

### 2-1. config 파일 수정하기

- _ base _의 dataset config 파일에서 data root를 원하는 폴더로 수정한다. 

  ```python
  python test.py "config파일 경로" "point pillars pth 경로" "YoloP pth 경로" --out "pkl파일 경로" --data-root "1-1의 폴더 경로"
  
  """
  EX)
  python test.py /opt/ml/final-project-level3-cv-08/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py /opt/ml/mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth /opt/ml/final-project-level3-cv-08/mmdetection3d/tools/model_best_train_lane_only.pth --out /opt/ml/pointpillar_eval_no_gt/result.pkl --eval mAP --data-root /opt/ml/kitti_0101
  """
  ```

   