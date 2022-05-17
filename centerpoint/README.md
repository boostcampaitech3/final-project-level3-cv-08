# mmdetection3d

### Centerpoint 사용법 (for nuscene)

- Preparation

  - Step1. nuscene dataset을 사용하기 위한 pkl 파일 생성

    ```python
    python mmdetection3d/tools/create_data.py '데이터셋 이름' --root-path '데이터셋 경로' --version '데이터셋 버전' --out-dir '결과 파일 저장 경로' --extra-tag '데이터셋 이름'
    
    """
    ex)
    python mmdetection3d/tools/create_data.py nuscenes --root-path /opt/ml/nuscene --version v1.0-mini --out-dir /opt/ml/nuscene --extra-tag nuscenes
    """
    ```

  - Step2. config 파일 수정

    ```python
    """
    mmdetection3d/configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py
    """
    
    data_root = 'nuscene 데이터 경로'
    # ex) 'opt/ml/nuscene/'
    ```

    ```python
    """
    mmdetection3d/configs/_base_/datasets/nus-3d.py
    """
    
    data_root = 'nuscene 데이터 경로'
    # ex) 'opt/ml/nuscene/'
    ```

  - Step3. download .pkl file

    [여기](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/centerpoint)에서 아래 빨간색으로 표시한 weight file 다운로드

    ![image-20220517194259563](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220517194259563.png)

  - Step4. 현재 모델 pth랑 모델 구조랑 조금 다른 부분이 있다. (point pillar 구조가 바뀌었는데 pth파일이 update가 안되어 있다고 한다.)

    ```python
    """
    mmdetection3d/mmdet3d/models/detectors/centerpoint.py
    """
    
    def extract_pts_feat(self, pts, img_feats, img_metas):
            """Extract features of points."""
            if not self.with_pts_bbox:
                return None
                
            # 이 부부 추가 
            # 문제점이 point의 채널이 원래는 x,y,z,r으로 4개인데, 뭐가 하나 더 추가되어서 5개의 채널을 갖는 것 같다. 그래서 마지막 채널 삭제해줘서 에러는 발생하진 않지만, 이렇게 해도 되는지는 확인이 필요하다. 
            for i in range(len(pts)):
                pts[i] = pts[i][:, :4]
            
            voxels, num_points, coors = self.voxelize(pts)
            voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
            batch_size = coors[-1, 0] + 1
            x = self.pts_middle_encoder(voxel_features, coors, batch_size)
            x = self.pts_backbone(x)
            if self.with_pts_neck:
                x = self.pts_neck(x)
            return x
    
        
    """
    mmdetection3d/configs/_base_/models/centerpoint_02pillar_second_secfpn_nus.py
    """
    model = dict(
        type='CenterPoint',
        pts_voxel_layer=dict(
            max_num_points=20, voxel_size=voxel_size, max_voxels=(30000, 40000)),
        pts_voxel_encoder=dict(
            type='PillarFeatureNet',
            in_channels=4, # 이 부분도 4채널로 맞춰줘야 함
            feat_channels=[64],
            with_distance=False,
            voxel_size=(0.2, 0.2, 8),
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
            legacy=False),
    ```

    

- Run

  ```python
  python mmdetection3d/tools/test.py  'config파일 경로' 'pth파일 경로' --out '결과 pkl파일 경로' --fuse-conv-bn '얘는 옵션인데, 이렇게 돌리면 좀 더 빠르다.'
  
  """
  ex)
  python test.py /opt/ml/mmdetection3d/configs/centerpoint/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus.py /opt/ml/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_20200930_103722-3bb135f2.pth --out /opt/ml/mmdetection3d/work_dir/result.pkl --fuse-conv-bn
  """
  ```

  