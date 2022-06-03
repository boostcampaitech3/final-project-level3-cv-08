# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp

import time
from PIL import Image
import mmcv
import torch
import numpy as np
from mmcv.image import tensor2imgs
import cv2
import os

from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)

from utils.utils import lidar2Bev, Sort, detection2Bev, bevPoints, drawBbox, transformImg
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from ..core.evaluation.kitti_utils.rotate_iou import rotate_iou_gpu_eval

def single_gpu_test(model,
                    model_2d,
                    data_loader,
                    args,
                    cfg,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3
                    ):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool, optional): Whether to save viualization results.
            Default: True.
        out_dir (str, optional): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    model_2d.eval()
    names = model_2d.module.names if hasattr(model_2d, 'module') else model_2d.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    results = []
    bev_results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       centerpoint_threshold=args.centerpoint_threshold) #create instance of the SORT tracker
    
    if not os.path.exists('/opt/ml/tracking'):
        os.makedirs('/opt/ml/tracking')
    with open('/opt/ml/tracking/tracking.txt', 'w') as tracking_file:
        print(f'if using cv2 to draw points -> x,y order : cv2.line(..., pt1=(x,y), pt2=(x,y), ...)', file=tracking_file)
        print(f'frame, tracking_id, x(garo), y(sero), class_id', file=tracking_file)
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)

            # 모든 lidar BEV view
            velodyne_path = dataset[i]['img_metas'][0].data['pts_filename']
            top, density_image = lidar2Bev(velodyne_path)
            
            # 모든 detection BEV view
            # total_det는 n x 7 numpy array -> x, y, rot, h, w, score, cls_id
            total_det = detection2Bev(result[0], i)

            # tracking
            trackers = mot_tracker.update(total_det)
            # x,y,rot,h,w -> x1,y1,x2,y2,x3,y3,x4,y4

            rotated_points= bevPoints(trackers)

            # draw bbox on bev lidar points
            density_image = drawBbox(rotated_points, trackers, density_image, i, tracking_file)

            cv2.imwrite(f'/opt/ml/images/3D/image{i:06d}.png', density_image)

            """
            YOLOP start
            """
            img_path = velodyne_path.replace('velodyne', 'image_2').replace('bin', 'png')
            img = Image.open(img_path).resize((640, 384))
            img_det = np.array(img)
            

            if torch.cuda.is_available():
                img = transformImg(img).unsqueeze(0).cuda()
            else:
                img = transformImg(img).unsqueeze(0).cpu()


            with torch.no_grad():
                det_out, da_seg_out, ll_seg_out = model_2d(img)

            inf_out, _ = det_out
            det_pred = non_max_suppression(inf_out, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
            det=det_pred[0]

            _, _, height, width = img.shape
            h,w,_=img_det.shape

            _, da_seg_out = torch.max(da_seg_out, 1)
            da_seg_out = da_seg_out.int().squeeze().cpu().numpy()

            _, ll_seg_out = torch.max(ll_seg_out, 1)
            ll_seg_out = ll_seg_out.int().squeeze().cpu().numpy()
            

            img_det = show_seg_result(img_det, (da_seg_out, ll_seg_out), _, _, is_demo=True)
            if len(det):
                det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
                for *xyxy,conf,cls in reversed(det):
                    label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
            
            
            cv2.imwrite(f'/opt/ml/images/2D/image{i:06d}.png', img_det)

            results.extend(result)
            bev_results.extend(total_det)
            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()
    return results, bev_results
