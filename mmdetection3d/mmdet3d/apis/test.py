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
import copy

from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)

from utils.utils import lidar2Bev, Sort, SortCustom, detection2Bev, from_latlon, bevPoints, drawBbox, transformImg, bevPoints_tracking, load_kitti_tracking_label, load_kitti_tracking_calib, Cam2LidarBev_tracking, makeForecastDict, filterUpdatedIds, forecastTest
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from ..core.evaluation.kitti_utils.rotate_iou import rotate_iou_gpu_eval

import sys
sys.path.append("../tools/PECNet/utils/")
from social_utils_add import *


def single_gpu_test(model,
                    model_2d,
                    model_forecast,
                    hyper_params,
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

    
    label = load_kitti_tracking_label('/opt/ml/kitti1/label_0000.txt')
    calib = load_kitti_tracking_calib('/opt/ml/kitti1/testing/calib/calib_0000.txt')
    R_vc = calib['R0_rect']
    T_vc = calib['Tr_velo_to_cam']
    P_ = calib['P2']
    v2p_matrix = P_@R_vc@T_vc

    mot_tracker = SortCustom(v2p_matrix = v2p_matrix,
                            iou_threshold = 0.4,
                            max_age=args.max_age, 
                            min_hits=args.min_hits,
                            centerpoint_threshold=args.centerpoint_threshold) #create instance of the SORT tracker
    
    bbox_2d_txt = open('/opt/ml/bbox_2d.txt', 'w')
    bbox_3d_txt = open('/opt/ml/bbox_3d.txt', 'w')
    forecast_dict = {}
    filtered_updated_ids = []
    predicted_updated_ids = []
    prev_forecast = np.empty((0, 0, 0))
    oxt_dict = {}
    with open('/opt/ml/kitti_testing_13/oxt_0013.txt', 'r') as oxt_file:
        oxt_datas = oxt_file.readlines()

    if not os.path.exists('/opt/ml/tracking'):
        os.makedirs('/opt/ml/tracking')
    with open('/opt/ml/tracking/tracking.txt', 'w') as tracking_file:
        print(f'if using cv2 to draw points -> x,y order : cv2.line(..., pt1=(x,y), pt2=(x,y), ...)', file=tracking_file)
        print(f'frame, tracking_id, x(garo), y(sero), class_id', file=tracking_file)
        for i, data in enumerate(data_loader):
            """
            YOLOP start
            """
            velodyne_path = dataset[i]['img_metas'][0].data['pts_filename']
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
            # 원래 사이즈로 bbox 복원
            bbox_2d = np.array(det[det[:, 5]<=2].detach().cpu())*np.array([1242/640, 375/384, 1242/640, 375/384, 1, 1])

            _, _, height, width = img.shape
            h,w,_=img_det.shape

            _, da_seg_out = torch.max(da_seg_out, 1)
            da_seg_out = da_seg_out.int().squeeze().cpu().numpy()

            _, ll_seg_out = torch.max(ll_seg_out, 1)
            ll_seg_out = ll_seg_out.int().squeeze().cpu().numpy()

            img_det = show_seg_result(img_det, (da_seg_out, ll_seg_out), _, _, is_demo=True)
            if len(det):
                # det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
                det[:, :4] *= torch.tensor([1242/640, 375/384, 1242/640, 375/384]).cuda()
                for *xyxy,conf,cls in reversed(det):
                    label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img_det, label=label_det_pred, color=colors[int(cls)], line_thickness=2)
            
            cv2.imwrite(f'/opt/ml/images/2D_recursive_sort/image{i:06d}.png', img_det)

            """
            3D start
            """
            # 현재 차량의 utm_x, utm_y, yaw를 구한다.
            oxt_data = oxt_datas[i].rstrip().split(' ') 
            oxt_data = np.array(oxt_data, dtype=np.float32)
            utm_coord = from_latlon(oxt_data[0], oxt_data[1])
            utm_x, utm_y = utm_coord[0], utm_coord[1] 
            oxt_dict[i] = np.array([utm_x, utm_y, oxt_data[5]+0.5]) # utm_x, utm_y, yaw


            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            
            """
            """
            gt, name = Cam2LidarBev_tracking(calib, label, i)
            gt= np.array(gt)[:, [0, 1, 3, 2, 4]]
            gt_points = bevPoints_tracking(gt)
            """
            """
            
            # 모든 lidar BEV view
            velodyne_path = dataset[i]['img_metas'][0].data['pts_filename']
            top, density_image, points_filtered = lidar2Bev(velodyne_path)
            
            # 모든 detection BEV view
            # total_det는 n x 8 numpy array -> x, y, z, l, w, h, yaw, score, cls_id
            labels = result[0]
            indices = np.where(labels['scores_3d'] >= 0.6)
            total_det = np.concatenate((labels['boxes_3d'].tensor.numpy()[indices], labels['scores_3d'].numpy()[indices][:, np.newaxis], labels['labels_3d'].numpy()[indices][:, np.newaxis]), axis=1)
            # tracking
            trackers = mot_tracker.update(total_det, bbox_2d[:, :4], points_filtered)
            # x,y,rot,h,w -> x1,y1,x2,y2,x3,y3,x4,y4
            
            
            forecast_dict, total_updated_ids, predicted_updated_ids = makeForecastDict(trackers, forecast_dict, filtered_updated_ids, prev_forecast, predicted_updated_ids, oxt_dict, i)
            filtered_updated_ids, filtered_forecast_dict = filterUpdatedIds(forecast_dict, total_updated_ids)

            # print(trackers) -> x, y, rot, l, w, tracking_id, cls_id, score, updated_coordinate(x, y, z, w, l, h, yaw)
            rotated_points= bevPoints(trackers)
            rotated_points_detections = bevPoints(total_det[:, [0, 1, 6, 3, 4]])

            # draw bbox on bev lidar points
            density_image = drawBbox(rotated_points, trackers, rotated_points_detections, density_image, i, tracking_file)

            """
            """
            # for j, point in enumerate(gt_points):
            #     point = point.reshape(-1, 2).astype(np.int32)[:,::-1]
            #     density_image = cv2.polylines(density_image, [point], True, (0, 0, 255), thickness=1)
            #     x_max = point[:, 0].max()
            #     y_max = point[:, 1].max()
            #     cv2.putText(density_image, str(j)+name[j], (x_max, y_max), 0, 0.7, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            """
            """
            

            for track in trackers:
                print("%d, %d, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f"%(i, track[5], track[8], track[9], track[10], track[11], track[12], track[13], track[14]), file=bbox_3d_txt)
            
            for bbox in  bbox_2d[:, :4]:
                print("%d, %d, %d, %d, %d"%(i, bbox[0], bbox[1], bbox[2], bbox[3]), file=bbox_2d_txt)
            


            """
            forecast start
            """
            
            # print(forecast_dict[1])
            if len(filtered_forecast_dict)!=0:
                forecast_test_dataset = SocialDataset(filtered_forecast_dict, set_name="test", b_size=25, t_tresh=0, d_tresh=25, verbose=True)

                recovery = np.empty((0, 0, 0))
                for traj in forecast_test_dataset.trajectory_batches:
                    recovery = copy.deepcopy(traj[:, :1, :])
                    traj -= traj[:, :1, :]
                    traj *= (hyper_params["data_scale"]*10)

                device = 'cuda'
                prev_forecast, density_image = forecastTest(forecast_test_dataset, model_forecast, device, hyper_params, density_image, recovery, forecast_dict, filtered_updated_ids,oxt_dict,best_of_n = 1)
            cv2.imwrite(f'/opt/ml/images/3D_recursive_sort/image{i:06d}.png', density_image)
            results.extend(result)
            bev_results.extend(total_det)
            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()
        bbox_2d_txt.close()
        bbox_3d_txt.close()   
    return results, bev_results
