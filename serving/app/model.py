import io
from typing import List, Dict, Any

import albumentations
import albumentations.pytorch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from lib.config import cfg
from lib.models.YOLOP import MCnet, get_net
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box, show_seg_result
import time
import cv2



def get_model(model_path: str = "/opt/ml/final-project-level3-cv-08/YOLOP/runs/BddDataset/model_best_train_da_only.pth") -> MCnet:
    """Model을 가져옵니다"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_net(cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
    return model


def _transform_image(image: Image):
    transform = albumentations.Compose(
        [
            #albumentations.Resize(height=640, width=384),
            albumentations.Normalize(mean=(0.362, 0.404, 0.398), std=(0.236, 0.274, 0.225)),
            albumentations.pytorch.transforms.ToTensorV2(),
        ]
    )
    ori_size = (image.width, image.height)
    image = image.resize((640, 384))
    image = image.convert("RGB")
    image_array = np.array(image)
    return ori_size, image_array, transform(image=image_array)["image"].unsqueeze(0)


def predict_from_image_byte(model: MCnet, image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return predict_from_image(model, image)

def predict_from_image(model: MCnet, image: Image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half = device.type != 'cpu'
    if half:
        model.half()
    model.eval()
    ls = []
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
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
