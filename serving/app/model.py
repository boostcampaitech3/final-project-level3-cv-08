import io
import os
from typing import List, Dict, Any

import albumentations
import albumentations.pytorch

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from lib_bdd.config import cfg
from lib_bdd.models.YOLOP import MCnet, get_net
from lib_bdd.core.general import non_max_suppression, scale_coords
from lib_bdd.core.function import AverageMeter
from lib_bdd.utils import plot_one_box, show_seg_result, time_synchronized, show_seg_result_video
import time
import cv2
import torch.backends.cudnn as cudnn
from lib_bdd.dataset import LoadImages

import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np
import sys
import glob
import copy
import subprocess as sp
import shlex


from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results

import yaml

import torchvision.transforms as transforms
normalize = transforms.Normalize(
        #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # kitti 
        mean=[0.362, 0.404, 0.398], std=[0.236, 0.274, 0.306]
    )

transform=transforms.Compose([
            #transforms.Resize((640, 384)),
            transforms.ToTensor(),
            normalize,
        ])

cfg_track = get_config(config_file="data/deep_sort.yaml")
'''with open("data/deep_sort.yaml", 'r') as fo:
            cfg_track.update(yaml.safe_load(fo.read()))'''
deepsort_model = build_tracker(cfg_track, use_cuda=True)
deepsort_car_model = build_tracker(cfg_track, use_cuda=True, is_car=True)


def get_model(model_path: str = "data/model_best_train_lane_only.pth") -> MCnet:
    """Model을 가져옵니다"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_net(cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
    return model


def _transform_image(image: Image):
    transform = albumentations.Compose(
        [
            #albumentations.Resize(height=720, width=1280),
            albumentations.Normalize(mean=(0.362, 0.404, 0.398), std=(0.236, 0.274, 0.225)),
            albumentations.pytorch.transforms.ToTensorV2(),
        ]
    )
    ori_size = (image.width, image.height)
    ori_img = image.convert("RGB")
    ori_img = ori_img.resize((1280, 720))
    ori_img = np.array(ori_img) #1280, 720
    image = image.resize((640, 384))
    image = image.convert("RGB")
    image_array = np.array(image) #640, 384
    return ori_size, ori_img, transform(image=image_array)["image"].unsqueeze(0)


def predict_from_image_byte(model: MCnet, image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return predict_from_image(model, image)

def predict_from_image(model: MCnet, image: Image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half = device.type != 'cpu'
    if half:
        model.half()
    ls = []
    names = model.module.names if hasattr(model, 'module') else model.names
    model.eval()
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    with torch.no_grad():
        ori_size, ori_img, transformed_image = _transform_image(image)
        #img_det = cv2.resize(ori_img, (640, 384))
        transformed_image = transformed_image.to(device)
        transformed_image = transformed_image.half() if half else transformed_image.float()
        if transformed_image.ndimension() == 3:
            transformed_image = transformed_image.unsqueeze(0)
        start = time.time()
        outputs = model(transformed_image)
        model_end = time.time()
        det_out, da_seg_out, ll_seg_out = outputs
        inf_out, _ = det_out
        det_pred = non_max_suppression(inf_out, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
        det = det_pred[0]
        _, da_seg_out = torch.max(da_seg_out, 1)
        da_seg_out = da_seg_out.int().squeeze().cpu().numpy()

        _, ll_seg_out = torch.max(ll_seg_out, 1)
        ll_seg_out = ll_seg_out.int().squeeze().cpu().numpy()

        img_det = show_seg_result(ori_img, (da_seg_out, ll_seg_out), _, _, is_demo=True)
        if len(det):#transformed_image.shape[2:]
            det[:,:4] = scale_coords(transformed_image.shape[2:],det[:,:4],img_det.shape).round() # 가로 세로가 같은 비율로 줄었기 때문에 다른 크기의 사진이라도 같은 비율이면 잘 나오지만 다른 비율이면 모든 연산이 끝난 후 resize를 해줘야한다.
            for *xyxy,conf,cls in reversed(det):
                label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                xyxy[0] = xyxy[0]
                plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
        #img_det = cv2.resize(img_det, (ori_size[0],ori_size[1]), interpolation=cv2.INTER_LINEAR)
        img_det = img_det.tolist()
    ls.append(img_det)
    twoDend = time.time()
    model_time = model_end - start
    plot_time = twoDend - model_end

    return ls, model_time, plot_time

def predict_from_video_bytes(model: MCnet, video_bytes: bytes):
    return predict_from_video(model, "uploaded.mp4", video_bytes)

def predict_from_video(model: MCnet, video_path: str, video_bytes = None):
    iou_thres = 0.45
    conf_thres = 0.25
    result = []
    id_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half = device.type != 'cpu'
    if half:
        model.half()
    model.eval()

    ls = []
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    cudnn.benchmark = True  # set True to speed up constant image size inference
    if video_bytes == None:
        dataset = LoadImages(video_path, img_size=640)
    else:
        dataset = video_bytes
    bs = len(dataset)  # batch_size
    
    t0 = time.time()

    vid_path, vid_writer = None, None
    path, img, img_det, vid_cap,shapes = next(iter(dataset))
    height, width, _ = img_det.shape

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    save_path_codec = str('result/'+ video_path.split("/")[-1][:-4] + "_before.mp4") if dataset.mode != 'stream' else str("data/result/web.mp4")
    save_path = str('result/'+ video_path.split("/")[-1]) if dataset.mode != 'stream' else str("data/result/web.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path_codec, fourcc, 8, (width, height), True)
    for i, (path, img, img_det, vid_cap,shapes) in enumerate(dataset):
        img = cv2.resize(img, (640, 384))
        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        det_out, da_seg_out,ll_seg_out= model(img)
        t2 = time_synchronized()
        # if i == 0:
        #     print(det_out)
        inf_out, _ = det_out
        inf_time.update(t2-t1,img.size(0))

        # Apply NMS
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, classes=None, agnostic=False)
        t4 = time_synchronized()

        nms_time.update(t4-t3,img.size(0))
        det=det_pred[0]
        #Path(path).name
        #save_path = str('/opt/ml/server_disk' +'/'+ Path(path).name.split('.')[0] + '.webm') if dataset.mode != 'stream' else str('/opt/ml/server_disk' + '/' + "web.mp4")
        

        _, _, height, width = img.shape
        h,w,_=img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)

        
        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        # Lane line post-processing
        #ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
        #ll_seg_mask = connect_lane(ll_seg_mask)

        img_det, segmentation = show_seg_result_video(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)
        det_info = list()
        if len(det):
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
            for *xyxy,conf,cls in reversed(det):
                cls, conf = int(cls.item()), conf.item()
                xyxy_int = [xy.item() for xy in xyxy]
                if cls == 0 or cls == 1:
                    det_info.append([i, cls, (xyxy_int[0]+xyxy_int[2])/2, (xyxy_int[1]+xyxy_int[3])/2, xyxy_int[2]-xyxy_int[0], xyxy_int[3]-xyxy_int[1], conf])
                label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
            det_info = np.array(det_info)
            #print(det_info.shape)
            img_det, result, id_dict = track(i, det_info, img_det, result, id_dict, segmentation)
        out.write(img_det)
    out.release()

    command = f"ffmpeg -y -i {save_path_codec} -vcodec libx264 {save_path}"
    process = sp.Popen(shlex.split(command), stdout=sp.PIPE, stderr=sp.STDOUT, universal_newlines=True)

    for line in process.stdout:
        print(line)
    #process = sp.Popen(shlex.split(f'ffmpeg -y -i {save_path} pipe: -vcodec libx264 {save_path}'), stdin=sp.PIPE)    
    '''process = sp.Popen(['ffmpeg', '-y', '-i', '{}'.format(save_path),
                           '-vcodec', 'libx264', f'{save_path}'],
                          stdout=sp.PIPE, shell=True)'''
    # Close and flush stdin
    #process.stdin.close()

    # Wait for sub-process to finish
    process.wait()

    print('Results saved to %s' % Path('/opt/ml/server_disk'))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))

    return save_path, inf_time.avg

def track(idx_frame, now_data, im, results, id_dict, seg):
    start = time.time()
    #now_data = np.array(now_data)

    bbox_xywh = now_data[:, 2:6].astype(float)
    cls_conf = now_data[:, 6].astype(float)
    cls_ids = now_data[:, 1].astype(int)

    origin_bbox_xywh = copy.deepcopy(bbox_xywh)
    origin_cls_conf = copy.deepcopy(cls_conf)
    img = copy.deepcopy(im)

    # select person class
    mask = cls_ids == 0

    bbox_xywh_p = origin_bbox_xywh[mask]
    # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
    #bbox_xywh[:, 3:] *= 1.2
    cls_conf_p = origin_cls_conf[mask]
    # do tracking
    #outputs = self.deepsort.update(bbox_xywh_p, cls_conf_p, im)

    # select car class
    mask = cls_ids == 1

    bbox_xywh_car = origin_bbox_xywh[mask]
    # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
    #bbox_xywh[:, 3:] *= 1.2
    cls_conf_car = origin_cls_conf[mask]

    # do tracking
    outputs = deepsort_model.update(bbox_xywh_p, cls_conf_p, im)
    outputs_p = outputs.copy()
    outputs = deepsort_car_model.update(bbox_xywh_car, cls_conf_car, img)
    outputs_c = outputs.copy()
    if len(outputs_p) == 0:
        outputs = outputs_c
    elif len(outputs_c) == 0:
        outputs = outputs_p
    else:
        outputs = np.concatenate((outputs_p, outputs_c))    
    
    # segmentatino 경고 영역 정하기
    shapes = np.zeros_like(im, np.uint8)
    h, w = len(seg), len(seg[0])
    print(h, w)
    x1, x2, y1, y2 = h,0,w,0
    for row_i in range(int(3*h/4), h):
        for col_i in range(int(w/4), int(3*w/4)):
            if seg[row_i][col_i][1] == 255:
                if x1 > row_i:
                    x1 = row_i
                if y1 > col_i:
                    y1 = col_i
                if x2 < row_i:
                    x2 = row_i
                if y2 < col_i:
                    y2 = col_i
    #cv2.rectangle(shapes, (y1, x1), (y2, x2), (0, 255, 255), -1)



    # draw boxes for visualization
    if len(outputs) > 0:
        bbox_tlwh = []
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -1]
        im = draw_boxes(im, bbox_xyxy, identities)

        # prediction
        for bb_xyxy in bbox_xyxy:
            bbox_tlwh.append(deepsort_model._xyxy_to_tlwh(bb_xyxy))

        results.append((idx_frame, bbox_tlwh, identities))
        for i in range(len(identities)):
            if identities[i] not in id_dict.keys():
                id_dict[identities[i]] = []
            elif idx_frame - 2 > id_dict[identities[i]][-1][0]:
                id_dict[identities[i]] = []
            id_dict[identities[i]].append([idx_frame, int(bbox_tlwh[i][0]+bbox_tlwh[i][2]/2), int((int(bbox_tlwh[i][1])+int(bbox_tlwh[i][3])))])
    
    if len(results) >= 4:
        for key, value in id_dict.items():
            value = np.array(value)
            if len(value) >= 4:
                # 과거 경로
                for l in range(3):
                    cv2.line(im, value[-1-l-1, 1:], value[-1-l, 1:], (0, 0, 255), 5)
                change = value[-1, 1:] - value[-4, 1:]
                # 미래 경로
                dst = value[-1, 1:] + change
                cv2.line(im, value[-1, 1:], dst, (0, 255, 0), 5)
                if x1 <= dst[1] <= x2 and y1 <= dst[0] <= y2:
                    cv2.line(shapes, value[0, 1:], dst, (0, 0, 255), 100)
                del id_dict[key][0]
    mask = shapes.astype(bool)
    out = im.copy()
    out[mask] = cv2.addWeighted(im, 0.5, shapes, 1 - 0.5, 0)[mask]
    end = time.time()

    # logging
    print("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                    .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))
    return out, results, id_dict


def get_config(config_path: str = "../../assets/mask_task/config.yaml"):
    import yaml

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config
