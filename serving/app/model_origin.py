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


def get_model(model_path: str = "data/model_best_train_seg_only.pth") -> MCnet:
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


def predict_from_video(model: MCnet, video_path: str):
    iou_thres = 0.45
    conf_thres = 0.25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half = device.type != 'cpu'
    if half:
        model.half()
    model.eval()

    ls = []
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadImages(video_path, img_size=640)
    bs = len(dataset)  # batch_size
    
    t0 = time.time()

    vid_path, vid_writer = None, None

    inf_time = AverageMeter()
    nms_time = AverageMeter()

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
        save_path = str('result/'+ Path(path).name.split('.')[0] + '.webm') if dataset.mode != 'stream' else str("data/result/web.mp4")

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

        img_det = show_seg_result_video(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

        if len(det):
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
            for *xyxy,conf,cls in reversed(det):
                label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
        
        if dataset.mode == 'images':
            cv2.imwrite(save_path,img_det)

        elif dataset.mode == 'video':
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'VP90'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h,w,_=img_det.shape
                vid_writer = cv2.VideoWriter(save_path, int(cv2.VideoWriter_fourcc(*fourcc)), fps, (w, h))
            vid_writer.write(img_det)
        
        else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)  # 1 millisecond

    print('Results saved to %s' % Path('/opt/ml/server_disk'))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))

    return save_path, inf_time.avg


























    
    with torch.no_grad():
        ori_size, img_det, transformed_image = _transform_image(image)
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

        img_det = show_seg_result(img_det, (da_seg_out, ll_seg_out), _, _, is_demo=True)
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


def get_config(config_path: str = "../../assets/mask_task/config.yaml"):
    import yaml

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config
