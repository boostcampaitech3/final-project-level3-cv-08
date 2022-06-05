from __future__ import print_function
from __future__ import division
import torch
import time
import math
import os
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from tqdm import tqdm
import glob
import time
import argparse
import cv2
from filterpy.kalman import KalmanFilter

import torchvision.transforms as transforms
import copy

######################################kalman filter를 위한 import
from copy import deepcopy
from math import log, exp, sqrt
import sys
import numpy as np
from numpy import dot, zeros, eye, isscalar, shape
import numpy.linalg as linalg
from filterpy.stats import logpdf
from filterpy.common import pretty_str, reshape_z
######################################
from scipy.optimize import linear_sum_assignment
import time
np.random.seed(0)
class KalmanFilterCustom(KalmanFilter):
    def __init__(self, dim_x, dim_z, dim_u=0):
        super().__init__(dim_x, dim_z)

    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter.
        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.
        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
            If you pass in a value of H, z must be a column vector the
            of the correct size.
        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        if H is None:
            z = reshape_z(z, self.dim_z, self.x.ndim)
            H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # 각도가 pi 부근에서는 차이가 적어도 음수에서 양수로 또는 양수에서 음수로 넘어가면서 매우 큰 차이로 나타난다.
        # 이를 해결해주기 위한 코드
        # ex) -3.13 -> 3.13일 때, 실제로는 0.02 차이지만 값은 6.26 차이가 된다.
        if abs(self.y[2][0]) > np.pi:
            if self.y[2][0] > 0 :
                self.y[2][0] = abs(self.y[2][0]) - 2*np.pi
            else:
                self.y[2][0] = 2*np.pi - abs(self.y[2][0])

        # 100m 회전 반경을 가진 곡선도로에서 횡방향으로 영향을 받지 않기 위한 속도는 50~60km라고 한다. 
        # 1초에 회전할 수 있는 각도는 0.16rad이고, 10 frame으로 가져오면 0.016rad이 나온다. 
        # 만약 ego 차량이 우측으로 회전하고, 상대 차량이 좌측으로 회전할 때는 값이 더 클 것이다. 
        # 대략 2배로 계산하면 0.032값이 나오고, update가 바로 안된다고 했을 때, 0.1정도가 최대일 것이다. 
        # 
        if abs(self.y[2][0]) > np.pi/5:
            self.y[2][0] = 0.0
        # common subexpression for speed
        PHT = dot(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
    

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

# iou_batch -> 3D centerpoint batch
def centerPointBatch(detection, predict):
    """
    From SORT : computes center point distance between two bboxes in the form [x,y,w,l,rot]
    원래 2D에서는 IoU를 계산하지만, 3D BEV에서는 occlusion이 없기 때문에 IoU를 계산하기보다는
    center point끼리의 거리와 w, l를 고려하면 충분할 듯 싶다.
    (center point가 matching에 더 중요한지 w, l이 더 중요한지에 따라서 가중치를 좀 다르게 주어 합산한 값을 계산할 예정)
    (아직 어떤게 더 중요할지에 대해서는 생각을 좀 더 해봐야 할 것 같다.)
    (현재 이 알고리즘의 문제점으로 예상되는 것이 연이어서 오는 차량의 경우 center point가 좀 애매해질 수도 있을 것 같다.
     예를 들어, 앞 뒤 두대의 차량이 존재하고, 앞 차량이 앞으로 진행했을 때, 뒤 차량도 앞으로 진행할 것이고 이에 따라서
     center point사이의 거리가 이전 프레임의 앞 차량과 이후 프레임의 뒤 차량의 center point의 거리가 더 가까워질 수도 있을 것 같다.
     만약 이럴경우에 두 차량이 모두 승용차라고 했을 때, w,l은 똑같을 거기 때문에 상대적으로 중요도를 center point에 주는 것이
     합리적일 것 같다.
     그렇다고 w,l을 고려하지 않기에는 만약 앞 차량이 큰 트럭이고, 뒤 차량이 승용차일 경우에는 w,l로 구분을 해줘야 할 것 같다.
     근데 또 드는 생각은 w,l이 객체에 따라서 m단위이기 때문에 차이가 좀 날 거 같은데 상수배로 weight를 주는 것이 맞는가, 아니면
     class까지 고려해서 weight를 줘야하는건지 좀 애매한 것 같다.
     class까지 고려하는 함수를 작성해야 하나...)
    + class + 각도까지 고려해줄까하는데 이거는 먼저 위에꺼 실험하고 결과가 안좋으면 추가해보는 걸로

    !!!일단은 center point만으로 한 번 해보고 성능이 안좋으면 그때 생각해보자!!!

    """
    predict = np.expand_dims(predict, 0)
    detection = np.expand_dims(detection, 1)
    center_dist = ((detection[..., 0]-predict[..., 0])**2 + (detection[..., 1]-predict[..., 1])**2)**0.5
           
    return(center_dist)  

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x, y, z, l, w, h, rot, score, cls_id] and returns z in the form
    [x,y,rot,l,w] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  new_bbox = bbox[[0, 1, 6, 3, 4]] 
  return new_bbox.reshape((-1, 1))

def convert_x_to_bbox(x, score=None):
  """
  Takes a bounding box in the centre form [x,y,rot,w,l, dx, dy, drot] and returns it in the form
    [x,y,rot, w, l] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  if(score==None):
    return np.array([x[0], x[1], x[2], x[3], x[4]]).reshape((1,-1))
  else:
    return np.array([x[0], x[1], x[2], x[3], x[4], score]).reshape((1,6))



class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.cls_id = bbox[8]
        self.score = bbox[7]
        self.value = bbox[:7]
        self.kf = KalmanFilterCustom(dim_x=8, dim_z=5)
        self.kf.F = np.array([[1,0,0,0,0,1,0,0],
                            [0,1,0,0,0,0,1,0],
                            [0,0,1,0,0,0,0,1],
                            [0,0,0,1,0,0,0,0],  
                            [0,0,0,0,1,0,0,0],
                            [0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0],
                            [0,0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.R[2, 2] *= 10.
        self.kf.P[5:,5:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[5:,5:] *= 0.01

        self.kf.x[:5] = convert_bbox_to_z(bbox)
        self.time_since_update = 0 # update가 되지 않았을 경우 1증가
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0 # update된 횟수
        self.hit_streak = 0 # 연속으로 update된 횟수
        self.age = 0 # 예측한 횟수

    def calib_rotation_y(self):
        if self.kf.x[2] > np.pi:
            self.kf.x[2] = -np.pi + (self.kf.x[2] - np.pi)
        elif self.kf.x[2] <= -np.pi:
            self.kf.x[2] = np.pi - (-np.pi - self.kf.x[2])
        
        return

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.calib_rotation_y()
            

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # if((self.kf.x[6]+self.kf.x[2])<=0):
        #     self.kf.x[6] *= 0.0
        self.kf.predict()

        # 각도가 pi보다 크거나, -pi보다 작거나 같을 때 생기는 문제점 해결  
        self.calib_rotation_y()

        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        # x, y, yaw, w, l
        # self.history[-1] : x, y, rot, w, l
        # self.value : x, y, z, w, l, h, rot, score, cls_id
        self.value[[0, 1, 6, 3, 4]] = self.history[-1]
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)[0], self.cls_id, self.score, self.value
        # x,y,rot,w,l / cls_id / score / updated coordinate(x, y, z, w, l, h, rot, score, cls_id) 

    

# 80km/h라고 하고 24fps로 영상을 가져온다고 했을 때, 
# 1 frame당 이동거리는 0.109m이다. 
# 0.109보다 조금 더 큰 값을 threshold로 준다. 
# 마주보고 달렸을 때를 생각하면 2배 정도 주는 것이 적당한 듯
def associate_detections_to_trackers(detections,trackers,centerpoint_threshold = 0.35):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,6),dtype=int)

    centerpoint_matrix = centerPointBatch(detections, trackers)
    if min(centerpoint_matrix.shape) > 0:
        a = (centerpoint_matrix < centerpoint_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(centerpoint_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)
    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(centerpoint_matrix[m[0], m[1]] > centerpoint_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
    def __init__(self, max_age=1, min_hits=3, centerpoint_threshold=0.35):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.centerpoint_threshold = centerpoint_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 9))):
        """
        Params:
        dets - a numpy array of detections in the format [[x,y,rot,w,l,score],[x,y,rot,w,l,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 6)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.centerpoint_threshold)
        # 3D와 tracking으로 predict된 객체와 matching 진행
        # matched           : match된 detection과 prediction
        # unmatched_dets    : match가 되지 않은 detections
        # unmatched_trks    : match가 되지 않은 detections
        
        
        # update matched trackers with assigned detections
        for m in matched:
            # print("matched_trks : ", self.trackers[m[1]].id)
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for . detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d, cls_id, cls_score, updated_coord = trk.get_state()
            
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1], [cls_id], [cls_score], updated_coord)).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,7))

#############################################################################################################################################


def IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

def IoUConfusionMatrix(bbox_2d, bbox_3d):
    """
    bbox_3d : 이미지 pixel좌표계의 3d bbox (8 corners) -> (N, 2, 8) -> 8개의 꼭지점 좌표
    bbox_2d : 이미지 pixel좌표계의 2d bbox (4 corners) -> (N, 4)    -> 2개의 꼭지점 좌표 (좌상단, 우하단)
    """
    four_edge_3ds = []
    for bbox_3d in bbox_3d:
        four_edge_3d = [bbox_3d[0, :].min(), bbox_3d[1, :].min(), bbox_3d[0, :].max(), bbox_3d[1, :].max()]
        four_edge_3ds.append(four_edge_3d)
    four_edge_3ds = np.array(four_edge_3ds)
    iou_matrix = np.zeros((bbox_2d.shape[0], four_edge_3ds.shape[0]))

    for i, bbox1 in enumerate(bbox_2d):
        for j, bbox2 in enumerate(four_edge_3ds):
            iou_matrix[i, j] = IoU(bbox1, bbox2)
    
    return iou_matrix

def match2D3D(bbox_2d, bbox_3d, iou_threshold_2d3d):
    """
    bbox_3d : 이미지 pixel좌표계의 3d bbox (8 corners) -> (N, 2, 8) -> 8개의 꼭지점 좌표
    bbox_2d : 이미지 pixel좌표계의 2d bbox (4 corners) -> (N, 4)    -> 2개의 꼭지점 좌표 (좌상단, 우하단)
    return
    matches_2d3d            : match된 2D, 3D index
    unmatched_indices_2d    : match 안 된 2D index
    unmatched_indices_3d    : match 안 된 3D index
    iou_matrix              : iou matrix
    """
    iou_matrix           = IoUConfusionMatrix(bbox_2d, bbox_3d)
    x, y                 = linear_sum_assignment(1-iou_matrix)
    
    matched_indices      = np.array(list(zip(x,y)))
    unmatched_indices_3d = []
    unmatched_indices_2d = []
    matches_2d3d = []
    

    if matched_indices.shape[0]==0:
        matched_indices = np.empty((0, 2))

    for i in range(bbox_2d.shape[0]):
        if i not in matched_indices[:, 0]:
            unmatched_indices_2d.append(i)
    
    for i in range(bbox_3d.shape[0]):
        if i not in matched_indices[:, 1]:
            unmatched_indices_3d.append(i)

    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < iou_threshold_2d3d):
            unmatched_indices_2d.append(m[0])
            unmatched_indices_3d.append(m[1])
        else:
            matches_2d3d.append(m.reshape(1,2))
    if len(matches_2d3d)!=0:
        matches_2d3d = np.concatenate(matches_2d3d)
    else:
        matches_2d3d = np.empty((0, 2))
    return matches_2d3d, np.array(unmatched_indices_2d), np.array(unmatched_indices_3d), iou_matrix

def toPixelCoord(bbox_3d, v2p_matrix):
    """
    bbox_3d     : (N, 3, 8) -> 8개의 꼭지점 좌표
    v2p_matrix  : (3, 4)    -> lidar coordinate to pixel coordinate
    return      : (N, 2, 8) -> 8개의 픽셀 좌표계로 표현된 꼭지점 좌표
    """
    bbox_3d_xyz_all = np.ones((bbox_3d.shape[0], 4, 8))
    bbox_3d_xyz_all[:, :3, :] = bbox_3d
    bbox_3d_xy_all = []
    for bbox_3d_xyz in bbox_3d_xyz_all:
        bbox_3d_xy = v2p_matrix@bbox_3d_xyz
        bbox_3d_xy = bbox_3d_xy / bbox_3d_xy[2, :].reshape(1, -1)
        bbox_3d_xy = bbox_3d_xy[:2, :][np.newaxis, :, :]
        bbox_3d_xy_all.append(bbox_3d_xy)
   
    return np.concatenate(bbox_3d_xy_all, axis=0)

def center2Edge(bbox_3d):
    """
    bbox_3d : (N, 9) -> (x,y,z,l,w,h,rot,score,cls)
    return : (N, 3, 8) -> 8개의 꼭지점 좌표
    """
    hwl              = bbox_3d[:, [5, 4, 3]]
    rotation_y_lidar = bbox_3d[:, 6]
    xyz_lidar        = bbox_3d[:, :3]
    bbox_edges       = []

    for i, (h,w,l) in enumerate(hwl):
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet\
                    [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
                    [0.0, 0.0, 0.0, 0.0, h, h, h, h]])

        yaw = rotation_y_lidar[i]
        rotMat = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], 
                        [np.sin(yaw), np.cos(yaw), 0.0], 
                        [0.0, 0.0, 1.0]])
        
        bbox_edges.append((np.dot(rotMat, trackletBox) + xyz_lidar[i].reshape(-1, 1))[np.newaxis, :, :])
    bbox_edges = np.concatenate(bbox_edges, axis=0)
    return bbox_edges

def associate_2d_detections_to_untracked_trackers(unmatched_dets, untracked_trks, v2p_matrix, iou_threshold_2d3d):
    if untracked_trks.shape[0]!=0:
        edge_bbox_3d = center2Edge(bbox_3d=untracked_trks)
        trks_pixel_3d = toPixelCoord(bbox_3d=edge_bbox_3d, v2p_matrix=v2p_matrix)
        matches_2d3d, _, _ , iou_matrix= match2D3D(unmatched_dets, trks_pixel_3d, iou_threshold_2d3d)

        return matches_2d3d, iou_matrix
    else:
        return np.empty((0, 2)), np.empty((0, 0))


def filterPointsByZ(point_in_2d_bbox, bbox_3d):
    """
    z축의 값으로 point를 필터링한다.

    point_in_2d_bbox    : (N, 4)
    bbox_3d             : (3, 8)
    """
    z_low, z_high = bbox_3d[2, 0], bbox_3d[2, 4]

    return point_in_2d_bbox[(point_in_2d_bbox[:, 2]>=z_low)&(point_in_2d_bbox[:, 2]<=z_high)]


def match2DBboxPoints(bbox_2d, points):
    """
    2D bbox와 lidar point를 matching한다.    

    bbox_2d     : (4, )
    points      : (K, 2)

    return
    matched_points : (N, ) -> N개의 match된 point index
    """
    bbox_2d = bbox_2d[np.newaxis, :]
    points = points.repeat(2, axis=0).reshape(-1, 4)
    calculated_matrix = bbox_2d-points
    matched_points = np.where((calculated_matrix[:, 0]<=0) & (calculated_matrix[:, 2] >= 0) 
                        & (calculated_matrix[:, 1]<=0) & (calculated_matrix[:, 3]>=0))  

    return matched_points[0]


def matchPoints(points, bbox_2d, v2p_matrix):
    """
    2D bbox안에 match되는 점들을 구한다.

    velodyne_path  : bin파일 경로
    image          : numpy array로 된 이미지
    bbox_2d        : (4, ) 2d bbox
    v2p_matrix     : (3, 4) velodyne to pixel coord matrix

    return
    matched_points : (N, ) -> N개의 match된 point index
    pixel_points   : (K, 2) -> pixel좌표계로 표현된 lidar points
    """
    pixel_points = points@v2p_matrix.T
    pixel_points = (pixel_points[:, :2] / pixel_points[:, 2].reshape(-1, 1)) # (K, 2)

    matched_points = match2DBboxPoints(bbox_2d, pixel_points)
    return matched_points, pixel_points

def point_in_quadrilateral(pt_x, pt_y, corners):
    """
    네개의 꼭지점으로 이루어진 다각형 안에 점이 속하는지 확인한다.

    corners     :   (4, 2) -> 4개의 꼭지점 좌표
    pt_x        :   (1, )  -> 확인하고자 하는 점의 x좌표
    pt_y        :   (1, )  -> 확인하고자 하는 점의 y좌표

    return
    boolean     :   True일 경우 내부의 점, False일 경우 외부의 점
    """
    ab0 = corners[1][0] - corners[0][0]
    ab1 = corners[1][1] - corners[0][1]

    ad0 = corners[3][0] - corners[0][0]
    ad1 = corners[3][1] - corners[0][1]

    ap0 = pt_x - corners[0][0]
    ap1 = pt_y - corners[0][1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1

    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0


def countPointsInBox(bbox_3d_xy, points_filtered):
    cnt = 0
    for point in points_filtered:
        cnt += point_in_quadrilateral(point[0], point[1], bbox_3d_xy)

    return cnt

def findMaxBbox(bbox_3d, points_filtered, shift_size):
    """
    4방향으로 shift하면서 최대치의 방향을 구한다. 

    bbox_3d             : (3, 8) -> 8개의 x,y,z 좌표
    points_filtered     : (N, 4) -> bbox_3d의 z좌표로 filtering된 2D bbox 내부 점들

    return
    bbox_3d_shifted     : (3, 8) -> 최대로 많은 점을 포함하고 있는 shift된 bbox
    boolean             : False일 경우 현재가 최대, True일 경우 shift된 bbox가 최대
    """
    bbox_3d_xy = bbox_3d[:2, :4].T # (4, 2)
    dxs = [shift_size, -shift_size, 0, 0, shift_size/2, shift_size, -shift_size, -shift_size]
    dys = [0, 0, shift_size, -shift_size, shift_size/2, -shift_size, shift_size, -shift_size]

    max_cnt = countPointsInBox(bbox_3d_xy, points_filtered)
    max_idx = -1
    for i, (dx, dy) in enumerate(zip(dxs, dys)):
        bbox_3d_xy_temp = bbox_3d_xy[:] + np.array([[dx, dy]])
        cnt = countPointsInBox(bbox_3d_xy_temp, points_filtered)
        if cnt > max_cnt:
            max_cnt = cnt
            max_idx = i
    if max_idx == -1:
        return bbox_3d, False, np.array([0, 0], dtype=np.float32)
    else:
        return bbox_3d + np.array([[dxs[max_idx]], [dys[max_idx]], [0]]), True, np.array([dxs[max_idx], dys[max_idx]], dtype=np.float32)


def bbox3dFourEdges(bbox_3d):
    """
    bbox_3d : 이미지 pixel좌표계의 3d bbox (8 corners) -> (N, 2, 8) -> 8개의 꼭지점 좌표
    """
    four_edge_3ds = []
    for bbox_3d in bbox_3d:
        four_edge_3d = [bbox_3d[0, :].min(), bbox_3d[1, :].min(), bbox_3d[0, :].max(), bbox_3d[1, :].max()]
        four_edge_3ds.append(four_edge_3d)
    four_edge_3ds = np.array(four_edge_3ds)

    return four_edge_3ds.squeeze()

    
def recursiveFindBestFit(bbox_edges, points_filtered, v2p_matrix, bbox_2d_check, prev_iou, shift_size):
    """
    3D bbox와 매칭된 2D bbox를 이용하여 recursive하게 3D bbox의 좌표를 보정

    bbox_edges      : (N, 3, 8)
    points_filtered : (K, 4)
    v2p_matrix      : (3, 4)
    prev_iou        : 초기 iou값
    shift_size      : 매 루프마다 이동할 거리 (m단위)
    
    return
    final_shifted   : 최종 이동 양 (x,y 방향 m단위)
    bbox_edges      : (N, 3, 8) -> 최종 보정된 8개의 꼭지점 좌표
    final_four_edges_2d : (4, ) -> 최종 3D에서 2D로 변환된 좌표
    """
    
    final_shifted       = np.array([0, 0], dtype=np.float32)
    final_four_edges_2d = bbox3dFourEdges(bbox_edges)
    while True:
        bbox_edges_new, shifted, total_shift = findMaxBbox(bbox_edges[0], points_filtered, shift_size)
        if shifted:
            bbox_edges_pixel    = toPixelCoord(bbox_edges_new[np.newaxis, :, :], v2p_matrix)
            four_edges_2d       = bbox3dFourEdges(bbox_edges_pixel)
            iou                 = IoU(four_edges_2d, bbox_2d_check)
            if iou > prev_iou:
                bbox_edges = bbox_edges_new[np.newaxis, :, :]
                final_shifted += total_shift
                final_four_edges_2d = four_edges_2d
                prev_iou = iou
            else:
                break
        else:
            break
    return final_shifted, bbox_edges, final_four_edges_2d

def filterDetections(detections):
    """
    detections : (N, 9)
    """
    centerpoint_matrix = centerPointBatch(detections, detections)
    for i in range(centerpoint_matrix.shape[0]):
        centerpoint_matrix[i, i:] = 1
    (idx_x, idx_y) = np.where(centerpoint_matrix < 0.1)
    filtered_detections = []
    for i, detection in enumerate(detections):
        if i in idx_x:
            continue
        filtered_detections.append(detection)
    
    return np.array(filtered_detections)

     
class SortCustom(object):
    def __init__(self, v2p_matrix, iou_threshold, max_age=1, min_hits=3, centerpoint_threshold=0.35):
        """
        Sets key parameters for SORT
        """
        self.iou_threshold = iou_threshold
        self.v2p_matrix = v2p_matrix
        self.max_age = max_age
        self.min_hits = min_hits
        self.centerpoint_threshold = centerpoint_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 9)), dets_pixel_2d=np.empty((0, 4)), points_filtered=np.empty((0, 4))):
        """
        Params:
        dets - a numpy array of detections in the format [[x,y,rot,w,l,score],[x,y,rot,w,l,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 6)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1


        """
        Kalman filter로 예측한 좌표와 detect한 좌표를 matching한다. 

        pos               : 예측한 좌표 (x, y, rot, w, l)
        dets              : (N, 9) -> (x, y, z, w, l, h, rot, score, cls_id)
        matched           : match된 detection과 prediction
        unmatched_dets    : match가 되지 않은 detections
        unmatched_trks    : match가 되지 않은 detections
        """
        dets = filterDetections(dets)
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 6))
        trks_total_coordinate = np.zeros((len(self.trackers), 7))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], 0]
            trks_total_coordinate[t][:] = self.trackers[t].value
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.centerpoint_threshold)
        """
        detection과 tracking이 match가 안 된 이유 가능성 크게 3 종류
            1. detection이 되지 않아 tracking과 matching이 안됐다. 
                1-1. tracking이 한 프레임밖에 안돼서 dx,dy가 업데이트가 안 됐을 때 -> 해결 방법이 마땅히 없다. 
                1-2. tracking이 여러 프레임 됐지만, detection이 안됐다. -> 제일 가능성 높음 -> 2D와 tracking을 matching하여 보완할 수 있을 것 같다. 
            2-1. tracking하던 객체가 화각을 벗어났다. -> 가능성이 높다.  
            2-2. 새로운 객체가 등장했다. -> 가능성 높다. 
            2-1, 2-2의 경우에는 그냥 새롭게 tracking에 추가하면 된다.  
            3. tracking의 dx, dy가 아직 update되지 않은 첫 번째 track 객체이다. -> 거리 범위를 좀 크게 줘서 가능성이 낮다.
        """

        """
        1-2번 가능성을 해결하기 위해서 unmatched 2D와 unmatched tracking matching
        """
        # find unmatched 2D
        edge_bbox_3d = center2Edge(bbox_3d=dets) # N, 3, 8
        dets_pixel_3d = toPixelCoord(bbox_3d=edge_bbox_3d, v2p_matrix=self.v2p_matrix)
        matches_2d3d, unmatched_indices_2d, unmatched_indices_3d, _ = match2D3D(dets_pixel_2d, dets_pixel_3d, self.iou_threshold)

        # match unmatched 3D tracking & unmatched 2D
        matches_2dtrks, iou_matrix = associate_2d_detections_to_untracked_trackers(dets_pixel_2d[unmatched_indices_2d.astype(np.int64)], trks_total_coordinate[unmatched_trks.astype(np.int64)], self.v2p_matrix, self.iou_threshold)

        # update 2d matched prediction with recursive 3D coordinate correction
        for m_2dtrk in matches_2dtrks:
            bbox_2d_check = dets_pixel_2d[unmatched_indices_2d[m_2dtrk[0]]]
            bbox_3d_check = self.trackers[unmatched_trks[m_2dtrk[1]]].value # x,y,z,w,l,h
            bbox_3d_check_edge = center2Edge(bbox_3d=bbox_3d_check[np.newaxis, :])
            matched_points, pixel_points = matchPoints(points_filtered, bbox_2d_check, self.v2p_matrix)
            points_filtered_matched = points_filtered[matched_points]
            points_filtered_matched = filterPointsByZ(points_filtered_matched, bbox_3d_check_edge[0])
            first_iou = iou_matrix[m_2dtrk[0], m_2dtrk[1]]
            final_shifted, final_bbox_edges, final_four_edges_2d = recursiveFindBestFit(bbox_3d_check_edge, 
                                                                                        points_filtered_matched, 
                                                                                        self.v2p_matrix, 
                                                                                        bbox_2d_check, 
                                                                                        first_iou, 
                                                                                        0.2)
            self.trackers[unmatched_trks[m_2dtrk[1]]].value += np.array([final_shifted[0], final_shifted[1], 0, 0, 0, 0, 0])
            self.trackers[unmatched_trks[m_2dtrk[1]]].update(self.trackers[unmatched_trks[m_2dtrk[1]]].value)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for . detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d, cls_id, cls_score, updated_coord = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1], [cls_id], [cls_score], updated_coord)).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,15))

############################################################################################################################################

def correctCoord(src_info, dst_info, dst_bbox_x, dst_bbox_y):
    """
    src info를 기준으로 dst_info에서 내 차량의 움직임을 제거한 bbox좌표 반환 (북쪽이 위쪽 방향)

    src info        : 내 차량의 기준 utm_x, utm_y, yaw (해당 id의 첫번째 인식된 객체 frame)
    dst_info        : 내 차량의 현재 utm_x, utm_y, yaw
    dst_bbox        : 보정할 bbox 좌표 x, y
    """
    src_utm = src_info[:2]

    dst_yaw = dst_info[2] - np.pi/2
    dst_utm = dst_info[:2]

    dst_rotation_mat = np.array([[np.cos(dst_yaw), -np.sin(dst_yaw)],
                            [np.sin(dst_yaw), np.cos(dst_yaw)]])

    dst_bbox_rot = np.array([dst_bbox_x, dst_bbox_y]).reshape(1, 2)@dst_rotation_mat.T # (1, 2)
    ego_moving = dst_utm[::-1] - src_utm[::-1] # (1, 2)

    corrected_bbox = dst_bbox_rot + ego_moving # (1, 2)

    return corrected_bbox[0][0], corrected_bbox[0][1]

def makeForecastDict(trackers, forecast_dict, prev_updated_ids, prev_forecast, predicted_updated_ids, oxt_dict, frame):
    """
    예측을 위한 dictionary를 만드는 함수

    trackers            : 현재 tracking된 객체 정보
    forecast_dict       : 예측을 위한 dictionary
    prev_updated_ids    : 이전 frame에서 예측에 사용된 ids
    prev_forecast       : 이전 frame에서 예측한 값들
    frame               : 현재 frame 번호

    return
    forecast_dict       : update된 예측을 위한 dictionary
    total_updated_ids   : 예측을 위해서 현재 tracking된 객체 ids와 현재는 tracking되지 않았지만, 이전 frame에서 예측한 ids
    predicted_updated_ids : 예측을 통해서 업데이트된 tracking ids
    """
    # tracking된 객체의 좌표 update
    cur_updated_ids = []
    #trackers_bev = center2ImageBev(trackers)
    
    for tracker in trackers:
        x,y = tracker[0], tracker[1]
        tracking_id = tracker[5]
        if tracking_id in forecast_dict.keys():
            src_frame = forecast_dict[tracking_id][0][1]
            corrected_x, corrected_y = correctCoord(oxt_dict[src_frame], oxt_dict[frame], x, y)
            forecast_dict[tracking_id] = np.append(forecast_dict[tracking_id], np.array([[tracking_id, frame, corrected_x, corrected_y]]), axis=0)
        else:
            corrected_x, corrected_y = correctCoord(oxt_dict[frame], oxt_dict[frame], x, y)
            forecast_dict[tracking_id] = np.array([[tracking_id, frame, corrected_x, corrected_y]])
        cur_updated_ids.append(tracking_id)
    
    total_updated_ids = copy.deepcopy(cur_updated_ids) 
    # 현재는 tracking 안됐지만, 이전에 예측된 값이라면 예측된 값으로 현재 frame 좌표를 채운다. 
    new_predicted_updated_ids = []
    for i, prev_id in enumerate(prev_updated_ids):
        if (prev_id not in cur_updated_ids) and (prev_id not in predicted_updated_ids):
            forecast_dict[prev_id] = np.append(forecast_dict[prev_id], np.array([[prev_id, frame, prev_forecast[i][0][0], prev_forecast[i][0][1]]]), axis=0)
            total_updated_ids.append(prev_id)
            new_predicted_updated_ids.append(prev_id)

    return forecast_dict, total_updated_ids, new_predicted_updated_ids


def filterUpdatedIds(forecast_dict, total_updated_ids):
    """
    update된 id 중에서 전체 길이가 8이상인 id만 filtering

    forecast_dict        : 예측을 위한 dictionary
    total_updated_ids    : update된 ids

    return
    filtered_updated_ids : 길이가 8이상인 ids
    """
    filtered_updated_ids = []
    filtered_forecast_dict = {}
    for id in total_updated_ids:
        if forecast_dict[id].shape[0] >= 8:
            filtered_updated_ids.append(id)
            filtered_forecast_dict[id] = forecast_dict[id][-8:]

    return filtered_updated_ids, filtered_forecast_dict


def recoverCoord(src_info, dst_info, points):
    """
    src info를 기준으로 dst_info에서 내 차량의 움직임을 제거한 bbox좌표 반환 (북쪽이 위쪽 방향)

    src info        : 내 차량의 기준 utm_x, utm_y, yaw (해당 id의 첫번째 인식된 객체 frame)
    dst_info        : 내 차량의 현재 utm_x, utm_y, yaw
    points          : 보정할 bbox 좌표 x, y (12, 2)
    """
    src_utm = src_info[:2]

    dst_yaw = np.pi/2 - dst_info[2]
    dst_utm = dst_info[:2]

    dst_rotation_mat = np.array([[np.cos(dst_yaw), -np.sin(dst_yaw)],
                            [np.sin(dst_yaw), np.cos(dst_yaw)]])

    ego_moving = dst_utm[::-1] - src_utm[::-1] # (1, 2)

    corrected_bbox = points - ego_moving # (12, 2)
    corrected_bbox = corrected_bbox@dst_rotation_mat.T  # (12, 2)
    
    return corrected_bbox


def addEgoMoving(forecast_dict, filtered_updated_ids, pf, oxt_dict):
    """
    bev로 그리기 위해서 좌표를 복원하는 과정

    pf      : 예측한 12프레임의 좌표 (N, 12, 2)
    """
    recovered_points_total = []
    for i, point_forecast in enumerate(pf):
        id = filtered_updated_ids[i]
        src_frame = forecast_dict[id][0][1]
        cur_frame = forecast_dict[id][-1][1]
        src_info = oxt_dict[src_frame]
        dst_info = oxt_dict[cur_frame]
        recovered_points = recoverCoord(src_info, dst_info, point_forecast) # (12, 2)
        recovered_points = center2ImageBev(recovered_points)
        recovered_points_total.append(recovered_points)

    recovered_points_total = np.array(recovered_points_total) # (N, 12, 2)
    return recovered_points_total

def forecastTest(test_dataset, model, device, hyper_params, density_image, recovery, forecast_dict, filtered_updated_ids,oxt_dict, best_of_n = 1):
    model.eval()
    assert best_of_n >= 1 and type(best_of_n) == int
    test_loss = 0
    with torch.no_grad():
        
        for i, (traj, mask, initial_pos) in enumerate(zip(test_dataset.trajectory_batches, test_dataset.mask_batches, test_dataset.initial_pos_batches)):
            traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)
            #x = traj[:, num:num+hyper_params["past_length"], :]
            # reshape the data
            x = traj[:, :hyper_params["past_length"], :]
            x = x.contiguous().view(-1, x.shape[1]*x.shape[2])
            x = x.to(device)

            for index in range(best_of_n):
                dest_recon = model.forward(x, initial_pos, device=device)
                dest_recon = dest_recon.cpu().numpy()

            best_guess_dest = dest_recon

            # back to torch land
            best_guess_dest = torch.DoubleTensor(best_guess_dest).to(device)

            # using the best guess for interpolation
            interpolated_future = model.predict(x, best_guess_dest, mask, initial_pos)
            interpolated_future = interpolated_future.cpu().numpy()
            best_guess_dest = best_guess_dest.cpu().numpy()
            
            # final overall prediction
            predicted_future = np.concatenate((interpolated_future, best_guess_dest), axis = 1)
            predicted_future = np.reshape(predicted_future, (-1, hyper_params["future_length"], 2))
            x /= (hyper_params["data_scale"]*10)
            x = np.array(x.detach().cpu()) + recovery.squeeze(1).repeat(8, axis=0).reshape(-1, 16)
            pf = predicted_future / (hyper_params["data_scale"]*10)
            pf += recovery[:, :1, :]

            recovered_x = addEgoMoving(forecast_dict, filtered_updated_ids, x.reshape(-1, hyper_params["past_length"], 2), oxt_dict) 
            recovered_pfs = addEgoMoving(forecast_dict, filtered_updated_ids, pf, oxt_dict)

            for j in range(len(x)):
                for k in range(8):
                    cv2.circle(density_image, (int(recovered_x[j][k][1]), int(recovered_x[j][k][0])), 1,(0,255,0), 5)
                for k in range(10):
                    cv2.circle(density_image, (int(recovered_pfs[j][k][1]), int(recovered_pfs[j][k][0])), 1,(0,0,255), 3)
        
        return pf, density_image




def load_kitti_calib(calib_file):
    if not os.path.exists(calib_file):
        print("calib file not exists!!!!!!")
        return None
    with open(calib_file) as f:
        lines = f.readlines()
        # 원래는 7개까지이지만, txt파일을 보면 마지막에 2 줄이 비어있다. 
        # 마지막 줄바꿈 문자까지 읽어오면 총 8줄이다. 
        # 맨 마지막 빈 줄은 파일의 끝을 의미
        assert (len(lines) == 7)

    # 맨 앞 글자 P0: 를 제외하고 뒤의 숫자만 읽어옴
    # strip()은 개행 문자를 제거하기 위해 사용
    obj = lines[0].strip().split(' ')[1:]
    #P0 = np.zeros((4, 4), dtype=np.float32)
    P0 = np.array(obj, dtype=np.float32).reshape(3, -1)
    #P0[3,3] = 1
    
    obj = lines[1].strip().split(' ')[1:]
    #P1 = np.zeros((4, 4), dtype=np.float32)
    P1 = np.array(obj, dtype=np.float32).reshape(3, -1)
    #P1[3,3] = 1
    
    obj = lines[2].strip().split(' ')[1:]
    #P2 = np.zeros((4, 4), dtype=np.float32)
    P2 = np.array(obj, dtype=np.float32).reshape(3, -1)
    #P2[3,3] = 1
    
    obj = lines[3].strip().split(' ')[1:]
    #P3 = np.zeros((4, 4), dtype=np.float32)
    P3 = np.array(obj, dtype=np.float32).reshape(3, -1)
    #P3[3,3] = 1
    
    obj = lines[4].strip().split(' ')[1:]
    R0_rect = np.zeros((4, 4), dtype=np.float32)
    R0_rect[:3,:3] = np.array(obj, dtype=np.float32).reshape(3, -1)
    R0_rect[3,3] = 1

    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.zeros((4, 4), dtype=np.float32)
    Tr_velo_to_cam[:3, :] = np.array(obj, dtype=np.float32).reshape(3, -1)
    Tr_velo_to_cam[3,3] = 1

    obj = lines[6].strip().split(' ')[1:]
    Tr_imu_to_velo = np.zeros((4, 4), dtype=np.float32)
    Tr_imu_to_velo[:3, :] = np.array(obj, dtype=np.float32).reshape(3, -1)
    Tr_imu_to_velo[3, 3] = 1

    return {'P0' : P0, 'P1' : P1, 'P2' : P2, 'P3' : P3, 'R0_rect' : R0_rect,
            'Tr_velo_to_cam' : Tr_velo_to_cam, 'Tr_imu_to_velo' : Tr_imu_to_velo}      

def load_kitti_label(label_file):
    classes = {'DontCare' : -1, 'Car' : 0, 'Pedestrian' : 1, 'Van' : 2, 'Cyclist' : 3}
    with open(label_file) as f:
        lines = f.readlines()
    frame = []
    tracking_id = []
    cls_id = []
    name = []
    truncated = []
    occluded = []
    alpha = []
    bbox = []
    dimensions = []
    location = []
    rotation_y = []
    difficulty = []
    score = []
    index = []
    idx = 0
    for line in lines:
        line = line.strip().split(' ')
        if line[2] not in classes.keys():
            continue
        frame.append(line[0])
        tracking_id.append(line[1])
        name.append(line[2])
        cls_id.append(classes[line[2]])
        truncated.append(float(line[3]))
        occluded.append(float(line[4]))
        alpha.append(float(line[5]))
        bbox.append([float(line[6]), float(line[7]), float(line[8]), float(line[9])])
        dimensions.append([float(line[10]), float(line[11]), float(line[12])])
        location.append([float(line[13]), float(line[14]), float(line[15])])

        if float(line[16]) <= -np.pi:
            rotation_y.append(np.pi)
        else:
            rotation_y.append(float(line[16]))
        difficulty.append(0)

        if name != 'DontCare':
            index.append(idx)
            idx += 1
        else:
            index.append(-1)

    frame = np.array(frame, dtype=np.int32)
    tracking_id = np.array(tracking_id, dtype=np.int32)
    cls_id = np.array(cls_id, dtype=np.int32)
    name = np.array(name, dtype='<U10')
    truncated = np.array(truncated, dtype=np.float32)
    occluded = np.array(occluded, dtype=np.float32)
    alpha = np.array(alpha, dtype=np.float32)
    bbox = np.array(bbox, dtype=np.float32)
    dimensions = np.array(dimensions, dtype=np.float32)
    location = np.array(location, dtype=np.float32)
    rotation_y = np.array(rotation_y, dtype=np.float32)
    difficulty = np.array(difficulty, dtype=np.float32)
    
    return {'frame' : frame, 'tracking_id': tracking_id, 'cls_id' : cls_id,
            'name' : name, 'truncated' : truncated,
            'occluded' : occluded, 'alpha' : alpha,
            'bbox' : bbox, 'dimensions' : dimensions,
            'location' : location, 'rotation_y' : rotation_y,
            'difficulty' : difficulty}

def bevPoints_tracking(trackers):
    xy = trackers[:, :2]
    rotation_y_lidar = trackers[:, 4]
    lw = trackers[:, 2:4]

    v = 0.1
    xrange = (0, 40.4)
    yrange = (-30, 30)
    xy_range = np.array([xrange[0], yrange[0]]).reshape(2, -1)

    W = math.ceil((xrange[1] - xrange[0]) / v)
    H = math.ceil((yrange[1] - yrange[0]) / v)
    X0, Xn = 0, W
    Y0, Yn = 0, H
    width = Yn - Y0
    height  = Xn - X0

    wh_range = np.array([height, width]).reshape(2, -1)
    
    rotated_points = []
    for i in range(xy.shape[0]):
        rotation_mat = np.array([[np.cos(rotation_y_lidar[i]), -np.sin(rotation_y_lidar[i])],
                            [np.sin(rotation_y_lidar[i]), np.cos(rotation_y_lidar[i])]])
        
        w = lw[i][1]/2
        l = lw[i][0]/2
        x_corners = [l, l, -l, -l]
        y_corners = [w, -w, -w, w]
        rotated_point = np.dot(rotation_mat, np.vstack([x_corners, y_corners]))
        rotated_point = wh_range - ((xy[i].repeat(4).reshape(2, -1) + rotated_point) - xy_range) / v
        rotated_point = rotated_point.T.reshape(8).astype(np.int32)
        rotated_points.append(rotated_point)
    return np.asarray(rotated_points)

def Cam2LidarBev_tracking(calib, label, frame):
    classes = ['Car', 'Pedestrian', 'Van', 'Cyclist']

    rect = calib['R0_rect']
    Tr_velo_to_cam = calib['Tr_velo_to_cam']
    rt_mat = np.linalg.inv(rect@Tr_velo_to_cam)
    # cam -> lidar


    name = label['name']

    xyz_base = label['location']
    xyz_base_shape = xyz_base.shape
    xyz = np.ones((xyz_base_shape[0], xyz_base_shape[1]+1))
    
    xyz[:, :3] = xyz_base
    hwl = label['dimensions']
    rotation_y = label['rotation_y']
    
    hwl_lidar = hwl
    xyz_lidar = xyz @ rt_mat.T

    rotation_y_lidar = -rotation_y - np.pi/2
    rotation_y_lidar = rotation_y_lidar - np.floor(rotation_y_lidar/(np.pi*2) + 1/2)*np.pi*2 
    # 각도가 pi/2 초과 pi 이하일 때, 값의 변환이 이상해지니 그 값에 filtering을 한다.
    # 라이다 포인트로 변환

    def filterIdx(classes, label, frame):
        idx = []
        for i in range(label['name'].shape[0]):
            if label['name'][i] in classes and label['frame'][i]==frame:
                idx.append(i)
        return idx

    
    
    filter_idx = filterIdx(classes, label, frame)
    
    hwl_lidar = hwl_lidar[filter_idx]
    xyz_lidar = xyz_lidar[filter_idx]
    rotation_y_lidar = rotation_y_lidar[filter_idx]
    name = name[filter_idx]
    
    gt = np.concatenate([xyz_lidar[:, :2], hwl_lidar[:, 1:3], rotation_y_lidar[:,np.newaxis]], axis=1)
    return gt, name

def load_kitti_tracking_calib(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()
        # 원래는 7개까지이지만, txt파일을 보면 마지막에 2 줄이 비어있다. 
        # 마지막 줄바꿈 문자까지 읽어오면 총 8줄이다. 
        # 맨 마지막 빈 줄은 파일의 끝을 의미
        assert (len(lines) == 7)

    # 맨 앞 글자 P0: 를 제외하고 뒤의 숫자만 읽어옴
    # strip()은 개행 문자를 제거하기 위해 사용
    obj = lines[0].strip().split(' ')[1:]
    #P0 = np.zeros((4, 4), dtype=np.float32)
    P0 = np.array(obj, dtype=np.float32).reshape(3, -1)
    #P0[3,3] = 1
    
    obj = lines[1].strip().split(' ')[1:]
    #P1 = np.zeros((4, 4), dtype=np.float32)
    P1 = np.array(obj, dtype=np.float32).reshape(3, -1)
    #P1[3,3] = 1
    
    obj = lines[2].strip().split(' ')[1:]
    #P2 = np.zeros((4, 4), dtype=np.float32)
    P2 = np.array(obj, dtype=np.float32).reshape(3, -1)
    #P2[3,3] = 1
    
    obj = lines[3].strip().split(' ')[1:]
    #P3 = np.zeros((4, 4), dtype=np.float32)
    P3 = np.array(obj, dtype=np.float32).reshape(3, -1)
    #P3[3,3] = 1
    
    obj = lines[4].strip().split(' ')[1:]
    R0_rect = np.zeros((4, 4), dtype=np.float32)
    R0_rect[:3,:3] = np.array(obj, dtype=np.float32).reshape(3, -1)
    R0_rect[3,3] = 1

    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.zeros((4, 4), dtype=np.float32)
    Tr_velo_to_cam[:3, :] = np.array(obj, dtype=np.float32).reshape(3, -1)
    Tr_velo_to_cam[3,3] = 1

    obj = lines[6].strip().split(' ')[1:]
    Tr_imu_to_velo = np.zeros((4, 4), dtype=np.float32)
    Tr_imu_to_velo[:3, :] = np.array(obj, dtype=np.float32).reshape(3, -1)
    Tr_imu_to_velo[3, 3] = 1
    return {'P0' : P0, 'P1' : P1, 'P2' : P2, 'P3' : P3, 'R0_rect' : R0_rect,
            'Tr_velo_to_cam' : Tr_velo_to_cam, 'Tr_imu_to_velo' : Tr_imu_to_velo}

def load_kitti_tracking_label(label_file):
    with open(label_file) as f:
        lines = f.readlines()

    frame = []
    name = []
    dimensions = []
    location = []
    rotation_y = []
    
    for line in lines:
        line = line.strip().split(' ')
        frame.append(int(line[0]))
        name.append(line[2])
        dimensions.append([float(line[10]), float(line[11]), float(line[12])])
        location.append([float(line[13]), float(line[14]), float(line[15])])
        rotation_y.append(float(line[16]))

    name = np.array(name, dtype='<U10')
    dimensions = np.array(dimensions, dtype=np.float32)
    location = np.array(location, dtype=np.float32)
    rotation_y = np.array(rotation_y, dtype=np.float32)
    return {'frame' : frame, 'name' : name, 'dimensions' : dimensions,
            'location' : location, 'rotation_y' : rotation_y,
            }

def detection2Bev(label, frame):
    classes = {0:'Pedestrian', 1:'Cyclist', 2:'Car'}

    # boxes_3d  : x,y,z,height, width, length
    label['boxes_3d'] = label['boxes_3d'].tensor.numpy()
    cls_id = label['labels_3d'].numpy()
    # cam -> lidar
    xyz_base = label['boxes_3d'][:, :3]
    xyz_base_shape = xyz_base.shape
    xyz = np.ones((xyz_base_shape[0], xyz_base_shape[1]+1))
    xyz[:, :3] = xyz_base
    hwl = label['boxes_3d'][:, 3:6]
    rotation_y = label['boxes_3d'][:, 6]    

    rotation_y_lidar = rotation_y
    # 각도가 pi/2 초과 pi 이하일 때, 값의 변환이 이상해지니 그 값에 filtering을 한다.
    # 라이다 포인트로 변환

    score = label['scores_3d'].numpy()
    
    detections = np.concatenate([xyz[:, :2], 
                                rotation_y_lidar[:,np.newaxis], 
                                hwl[:, 0:2], 
                                score[:, np.newaxis], 
                                cls_id[:, np.newaxis]], axis=1)
    return detections


def lidar2Bev(velodyne_path):
    vd = 0.4
    vh = 0.1
    vw = 0.1
    
    # xrange = (0, 40.4)
    # yrange = (-30, 30)
    xrange = (0, 70.4)
    yrange = (-40, 40)
    zrange = (-3, 1)

    W = math.ceil((xrange[1] - xrange[0]) / vw)
    H = math.ceil((yrange[1] - yrange[0]) / vh)
    D = math.ceil((zrange[1] - zrange[0]) / vd)
    
    X0, Xn = 0, W
    Y0, Yn = 0, H
    Z0, Zn = 0, D

    width  = Yn - Y0
    height   = Xn - X0
    channel = Zn - Z0  + 2


    if not os.path.exists(velodyne_path):
        print(f'{velodyne_path} not exists!!!!!!')
        return np.zeros((height, width, channel)), np.zeros((height, width)), np.empty((0, 4))

    lidar = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)

    def filter_points(lidar, xrange, yrange, zrange):
        lidar = lidar[:][(lidar[:,0] >= xrange[0]) & (lidar[:,0] < xrange[1])]
        lidar = lidar[:][(lidar[:,1] >= yrange[0]) & (lidar[:,1] < yrange[1])]
        lidar = lidar[:][(lidar[:,2] >= zrange[0]) & (lidar[:,2] < zrange[1])]
        
        return lidar
    start = time.time()
    lidar = filter_points(lidar, xrange, yrange, zrange)
    
    # 원래 voxel의 channel보다 2개를 더해주는데
    # 뒤에 top이랑 mask에서 용도가 나옴
    
    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    prs = lidar[:, 3]

    qxs=((pxs-xrange[0])/vw).astype(np.int32)
    qys=((pys-yrange[0])/vh).astype(np.int32)
    qzs=((pzs-zrange[0])/vd).astype(np.int32)
    # voxel index를 의미

    #print('height,width,channel=%d,%d,%d'%(height,width,channel))

    top = np.zeros(shape=(height,width,channel), dtype=np.float32)
    # top에는 z 좌표를 바닥이 0부터 시작하게 하여 양수만 저장
    # top의 맨 마지막 channel에 만약 해당 (heigh, width)상에 점이 없을 경우는 0, 있을 경우는 1을 더해줌으로써 점이 여러개면 그 값이 커진다. 
    # 맨 마지막 바로 전 channel에는 reflectance를 저장 (같은 (height, width)상의 점들 중 z값이 가장 큰 값의)

    mask = np.ones(shape=(height,width,channel-1), dtype=np.float32)* -5
    # mask에는 voxel 안에 점들 중 높이가 최대인 값(음수 상관 x)을 저장
    # -5로 초기화 하는 이유는 z축의 최소 범위가 -3이기 때문인듯 (-4도 됨, -3보다 더 작은 값이기만 하면 됨)
    # 맨 마지막 채널에는 같은 (height, widht)상의 점들 중 가장 큰 값을 저장

    top[-qxs, -qys, -1] += 1
    """
    for i in range(len(pxs)):
        top[-qxs[i], -qys[i], -1]= 1 + top[-qxs[i], -qys[i], -1]
        # top의 마지막 채널은 point가 (height, widht)상에 존재하면 1을 더하고 존재하지 않으면 더하지 않는다. 
        # 이렇게 점이 많으면 값이 커지고 이미지로 plot할 때 더 흰값을 갖게 된다.
        # 음수로 한 이유는
        #           z 증가하는 방향
        #           ^ 
        #           |_____> y 증가하는 방향
        #          / 
        #         /
        #        x 증가하는 방향
        #     차 머리 방향
        # 이렇게 되어 있기 사실상 우하단이 원점이라고 볼 수 있다. 
        # 이미지에서는 좌상단이 원점이므로 x와 y에 음수를 취해서 이미지를 뒤집어준것
        
        if pzs[i]>mask[-qxs[i], -qys[i],qzs[i]]:
            # mask에는 voxel마다 실제 높이 좌표가 가장 큰 값이 저장되어 있음 (음수도 가능)
            # 그 voxel내의 좌표보다 현재 좌표가 더 크다면
            top[-qxs[i], -qys[i], qzs[i]] = max(0,pzs[i]-zrange[0])
            # top은 voxel마다 양수로 변환한 좌표가 저장되어 있음
            # 현재 값이 더 크므로 갱신
            mask[-qxs[i], -qys[i],qzs[i]]=pzs[i]
            # 마찬가지로 갱신하되 음수여도 그냥 집어넣음
        if pzs[i]>mask[-qxs[i], -qys[i],-1]:
            # mask의 마지막 채널에는 해당 (height, width) 상에서 가장 큰 값이 저장되어 있음
            # 이 값보다 더 큰 값이 있다면
            mask[-qxs[i], -qys[i],-1]=pzs[i]
            # mask의 가장 큰 값 갱신
            top[-qxs[i], -qys[i], -2]=prs[i]
            # top의 뒤에서 두번째 channel에는 해당 값의 reflectance를 저장
    """

    top[:,:,-1] = np.log(top[:,:,-1]+1)/math.log(64)
    # 이건 normalize방식인데 논문에 나와있음
    # top의 마지막 채널에는 point가 많을 경우 큰값, 적을 경우 작은값, 없을 경우 0인데, 이 값을 normalize함
    # log를 취하니깐 log1=0이니깐 값이 없으면 0, 아니면 logn/log64
    # 이렇게 되면 점이 많은 곳과 적은 곳의 색 차이가 나타나게 됨
    
    # top_image = np.sum(top[:,:,:-1],axis=2)
    density_image = top[:,:,-1]
    density_image = density_image-np.min(density_image)
    # 음수를 양수로 바꿔주기 위해 최소 값을 뺌
    density_image = (density_image/np.max(density_image)*255).astype(np.uint8)
    # 최대값으로 나눠주어 0~1사이 값으로 만들고 255를 곱해주어 픽셀 값으로 만들어줌
    # top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)

    return top, density_image, lidar

def center2ImageBev(trackers):
    xy = trackers[:, :2]

    v = 0.1
    xrange = (0, 70.4)
    yrange = (-40, 40)

    xy_range = np.array([xrange[0], yrange[0]]).reshape(-1, 2)

    W = math.ceil((xrange[1] - xrange[0]) / v)
    H = math.ceil((yrange[1] - yrange[0]) / v)
    X0, Xn = 0, W
    Y0, Yn = 0, H
    width = Yn - Y0
    height  = Xn - X0

    wh_range = np.array([height, width]).reshape(-1, 2)

    trackers_bev = wh_range - (xy - xy_range)/v
    
    return trackers_bev    


def bevPoints(trackers):
    xy = trackers[:, :2]
    rotation_y_lidar = trackers[:, 2]
    lw = trackers[:, 3:5]

    v = 0.1
    # xrange = (0, 40.4)
    # yrange = (-30, 30)
    xrange = (0, 70.4)
    yrange = (-40, 40)
    xy_range = np.array([xrange[0], yrange[0]]).reshape(2, -1)

    W = math.ceil((xrange[1] - xrange[0]) / v)
    H = math.ceil((yrange[1] - yrange[0]) / v)
    X0, Xn = 0, W
    Y0, Yn = 0, H
    width = Yn - Y0
    height  = Xn - X0

    wh_range = np.array([height, width]).reshape(2, -1)
    
    rotated_points = []
    for i in range(xy.shape[0]):
        rotation_mat = np.array([[np.cos(rotation_y_lidar[i]), -np.sin(rotation_y_lidar[i])],
                            [np.sin(rotation_y_lidar[i]), np.cos(rotation_y_lidar[i])]])
        
        w = lw[i][1]/2
        l = lw[i][0]/2
        x_corners = [l, l, -l, -l]
        y_corners = [w, -w, -w, w]
        rotated_point = np.dot(rotation_mat, np.vstack([x_corners, y_corners]))
        rotated_point = wh_range - ((xy[i].repeat(4).reshape(2, -1) + rotated_point) - xy_range) / v
        rotated_point = rotated_point.T.reshape(8).astype(np.int32)
        rotated_points.append(rotated_point)
    return np.asarray(rotated_points)

def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def drawBbox(box3d, trackers, rotated_points_detections, density_image, frame, tracking_file):
    classes = {0 : 'Pedestrian', 1 : 'Cyclist', 2 : 'Car'}
    img_2d = density_image[np.newaxis, :, :].repeat(3, 0).transpose(1, 2, 0).astype(np.int32).copy()

    if trackers.shape[0]==0:
        return img_2d

    ids = trackers[:, 5].reshape(-1).astype(np.int32)
    scores = trackers[:, 7].reshape(-1).astype(np.float64)
    labels = trackers[:, 6].reshape(-1)
    

    # ground truth of detections
    for i in range(rotated_points_detections.shape[0]):
        color = (255, 0, 0)
        points = rotated_points_detections[i].reshape(-1, 2)
        points = points[:, ::-1]
        x_max = points[:,0].max()
        y_max = points[:,1].max()
        img_2d = cv2.polylines(img_2d, [points], True, color, thickness=3)
        img_2d = cv2.line(img_2d, ((points[0][0]+points[2][0])//2, (points[0][1]+points[2][1])//2), ((points[0][0]+points[2][0])//2, (points[0][1]+points[2][1])//2), color = color, thickness=5)
        # print('%d, %d, %d, %d, %d'%(frame, ids[i], (points[0b][0]+points[2][0])//2, (points[0][1]+points[2][1])//2, labels[i]), file=tracking_file)
        # cv2.putText(img_2d, classes[labels[i]]+str(ids[i])+' : ' + str(scores[i])[:4], (x_max, y_max), 0, 0.7, color, thickness=1, lineType=cv2.LINE_AA)
        # cv2.putText(img_2d, 'frame : ' + str(frame), (5, 30), 0, 0.7, color, thickness=1, lineType=cv2.LINE_AA)
    

    # detections updated with Kalmanfilter
    for i in range(box3d.shape[0]):
        color = compute_color_for_id(ids[i])
        points = box3d[i].reshape(-1, 2)
        points = points[:, ::-1]
        x_max = points[:,0].max()
        y_max = points[:,1].max()
        img_2d = cv2.polylines(img_2d, [points], True, color, thickness=2)
        img_2d = cv2.line(img_2d, ((points[0][0]+points[2][0])//2, (points[0][1]+points[2][1])//2), ((points[0][0]+points[2][0])//2, (points[0][1]+points[2][1])//2), color = color, thickness=4)
        print('%d, %d, %d, %d, %d'%(frame, ids[i], (points[0][0]+points[2][0])//2, (points[0][1]+points[2][1])//2, labels[i]), file=tracking_file)
        # cv2.putText(img_2d, classes[labels[i]]+str(ids[i])+' : ' + str(scores[i])[:4], (x_max, y_max), 0, 0.7, color, thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(img_2d, str(ids[i]), (x_max, y_max), 0, 0.7, color, thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(img_2d, 'frame : ' + str(frame), (5, 30), 0, 0.7, color, thickness=2, lineType=cv2.LINE_AA)
    
    
    return img_2d


######for YOLOP

transform=transforms.Compose([
            # transforms.Resize((384, 640)),
            transforms.ToTensor(),
            transforms.Normalize(
            #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # kitti 
            mean=[0.362, 0.404, 0.398], std=[0.236, 0.274, 0.306]
            )
        ])
def transformImg(img):
    return transform(img)
    



class OutOfRangeError(ValueError):
    pass


__all__ = ['to_latlon', 'from_latlon']

K0 = 0.9996

E = 0.00669438
E2 = E * E
E3 = E2 * E
E_P2 = E / (1 - E)

SQRT_E = np.sqrt(1 - E)
_E = (1 - SQRT_E) / (1 + SQRT_E)
_E2 = _E * _E
_E3 = _E2 * _E
_E4 = _E3 * _E
_E5 = _E4 * _E

M1 = (1 - E / 4 - 3 * E2 / 64 - 5 * E3 / 256)
M2 = (3 * E / 8 + 3 * E2 / 32 + 45 * E3 / 1024)
M3 = (15 * E2 / 256 + 45 * E3 / 1024)
M4 = (35 * E3 / 3072)

P2 = (3 / 2 * _E - 27 / 32 * _E3 + 269 / 512 * _E5)
P3 = (21 / 16 * _E2 - 55 / 32 * _E4)
P4 = (151 / 96 * _E3 - 417 / 128 * _E5)
P5 = (1097 / 512 * _E4)

R = 6378137

ZONE_LETTERS = "CDEFGHJKLMNPQRSTUVWXX"


def in_bounds(x, lower, upper, upper_strict=False):
    if upper_strict and True:
        return lower <= np.min(x) and np.max(x) < upper
    elif upper_strict and not True:
        return lower <= x < upper
    elif True:
        return lower <= np.min(x) and np.max(x) <= upper
    return lower <= x <= upper


def check_valid_zone(zone_number, zone_letter):
    if not 1 <= zone_number <= 60:
        raise OutOfRangeError('zone number out of range (must be between 1 and 60)')

    if zone_letter:
        zone_letter = zone_letter.upper()

        if not 'C' <= zone_letter <= 'X' or zone_letter in ['I', 'O']:
            raise OutOfRangeError('zone letter out of range (must be between C and X)')


def mixed_signs(x):
    return True and np.min(x) < 0 and np.max(x) >= 0


def negative(x):
    if True:
        return np.max(x) < 0
    return x < 0


def mod_angle(value):
    """Returns angle in radians to be between -pi and pi"""
    return (value + np.pi) % (2 * np.pi) - np.pi


def to_latlon(easting, northing, zone_number, zone_letter=None, northern=None, strict=True):
    """This function converts UTM coordinates to Latitude and Longitude

        Parameters
        ----------
        easting: int or NumPy array
            Easting value of UTM coordinates

        northing: int or NumPy array
            Northing value of UTM coordinates

        zone_number: int
            Zone number is represented with global map numbers of a UTM zone
            numbers map. For more information see utmzones [1]_

        zone_letter: str
            Zone letter can be represented as string values.  UTM zone
            designators can be seen in [1]_

        northern: bool
            You can set True or False to set this parameter. Default is None

        strict: bool
            Raise an OutOfRangeError if outside of bounds

        Returns
        -------
        latitude: float or NumPy array
            Latitude between 80 deg S and 84 deg N, e.g. (-80.0 to 84.0)

        longitude: float or NumPy array
            Longitude between 180 deg W and 180 deg E, e.g. (-180.0 to 180.0).


       .. _[1]: http://www.jaworski.ca/utmzones.htm

    """
    if not zone_letter and northern is None:
        raise ValueError('either zone_letter or northern needs to be set')

    elif zone_letter and northern is not None:
        raise ValueError('set either zone_letter or northern, but not both')

    if strict:
        if not in_bounds(easting, 100000, 1000000, upper_strict=True):
            raise OutOfRangeError('easting out of range (must be between 100,000 m and 999,999 m)')
        if not in_bounds(northing, 0, 10000000):
            raise OutOfRangeError('northing out of range (must be between 0 m and 10,000,000 m)')
    
    check_valid_zone(zone_number, zone_letter)
    
    if zone_letter:
        zone_letter = zone_letter.upper()
        northern = (zone_letter >= 'N')

    x = easting - 500000
    y = northing

    if not northern:
        y -= 10000000

    m = y / K0
    mu = m / (R * M1)

    p_rad = (mu +
             P2 * np.sin(2 * mu) +
             P3 * np.sin(4 * mu) +
             P4 * np.sin(6 * mu) +
             P5 * np.sin(8 * mu))

    p_sin = np.sin(p_rad)
    p_sin2 = p_sin * p_sin

    p_cos = np.cos(p_rad)

    p_tan = p_sin / p_cos
    p_tan2 = p_tan * p_tan
    p_tan4 = p_tan2 * p_tan2

    ep_sin = 1 - E * p_sin2
    ep_sin_sqrt = np.sqrt(1 - E * p_sin2)

    n = R / ep_sin_sqrt
    r = (1 - E) / ep_sin

    c = E_P2 * p_cos**2
    c2 = c * c

    d = x / (n * K0)
    d2 = d * d
    d3 = d2 * d
    d4 = d3 * d
    d5 = d4 * d
    d6 = d5 * d

    latitude = (p_rad - (p_tan / r) *
                (d2 / 2 -
                 d4 / 24 * (5 + 3 * p_tan2 + 10 * c - 4 * c2 - 9 * E_P2)) +
                 d6 / 720 * (61 + 90 * p_tan2 + 298 * c + 45 * p_tan4 - 252 * E_P2 - 3 * c2))

    longitude = (d -
                 d3 / 6 * (1 + 2 * p_tan2 + c) +
                 d5 / 120 * (5 - 2 * c + 28 * p_tan2 - 3 * c2 + 8 * E_P2 + 24 * p_tan4)) / p_cos

    longitude = mod_angle(longitude + np.radians(zone_number_to_central_longitude(zone_number)))

    return (np.degrees(latitude),
            np.degrees(longitude))


def from_latlon(latitude, longitude, force_zone_number=None, force_zone_letter=None):
    """This function converts Latitude and Longitude to UTM coordinate

        Parameters
        ----------
        latitude: float or NumPy array
            Latitude between 80 deg S and 84 deg N, e.g. (-80.0 to 84.0)

        longitude: float or NumPy array
            Longitude between 180 deg W and 180 deg E, e.g. (-180.0 to 180.0).

        force_zone_number: int
            Zone number is represented by global map numbers of an UTM zone
            numbers map. You may force conversion to be included within one
            UTM zone number.  For more information see utmzones [1]_

        force_zone_letter: str
            You may force conversion to be included within one UTM zone
            letter.  For more information see utmzones [1]_

        Returns
        -------
        easting: float or NumPy array
            Easting value of UTM coordinates

        northing: float or NumPy array
            Northing value of UTM coordinates

        zone_number: int
            Zone number is represented by global map numbers of a UTM zone
            numbers map. More information see utmzones [1]_

        zone_letter: str
            Zone letter is represented by a string value. UTM zone designators
            can be accessed in [1]_


       .. _[1]: http://www.jaworski.ca/utmzones.htm
    """
    if not in_bounds(latitude, -80, 84):
        raise OutOfRangeError('latitude out of range (must be between 80 deg S and 84 deg N)')
    if not in_bounds(longitude, -180, 180):
        raise OutOfRangeError('longitude out of range (must be between 180 deg W and 180 deg E)')
    if force_zone_number is not None:
        check_valid_zone(force_zone_number, force_zone_letter)

    lat_rad = np.radians(latitude)
    lat_sin = np.sin(lat_rad)
    lat_cos = np.cos(lat_rad)

    lat_tan = lat_sin / lat_cos
    lat_tan2 = lat_tan * lat_tan
    lat_tan4 = lat_tan2 * lat_tan2

    if force_zone_number is None:
        zone_number = latlon_to_zone_number(latitude, longitude)
    else:
        zone_number = force_zone_number

    if force_zone_letter is None:
        zone_letter = latitude_to_zone_letter(latitude)
    else:
        zone_letter = force_zone_letter

    lon_rad = np.radians(longitude)
    central_lon = zone_number_to_central_longitude(zone_number)
    central_lon_rad = np.radians(central_lon)

    n = R / np.sqrt(1 - E * lat_sin**2)
    c = E_P2 * lat_cos**2

    a = lat_cos * mod_angle(lon_rad - central_lon_rad)
    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    a5 = a4 * a
    a6 = a5 * a

    m = R * (M1 * lat_rad -
             M2 * np.sin(2 * lat_rad) +
             M3 * np.sin(4 * lat_rad) -
             M4 * np.sin(6 * lat_rad))

    easting = K0 * n * (a +
                        a3 / 6 * (1 - lat_tan2 + c) +
                        a5 / 120 * (5 - 18 * lat_tan2 + lat_tan4 + 72 * c - 58 * E_P2)) + 500000

    northing = K0 * (m + n * lat_tan * (a2 / 2 +
                                        a4 / 24 * (5 - lat_tan2 + 9 * c + 4 * c**2) +
                                        a6 / 720 * (61 - 58 * lat_tan2 + lat_tan4 + 600 * c - 330 * E_P2)))

    if mixed_signs(latitude):
        raise ValueError("latitudes must all have the same sign")
    elif negative(latitude):
        northing += 10000000

    return easting, northing, zone_number, zone_letter


def latitude_to_zone_letter(latitude):
    # If the input is a numpy array, just use the first element
    # User responsibility to make sure that all points are in one zone
    if True and isinstance(latitude, np.ndarray):
        latitude = latitude.flat[0]

    if -80 <= latitude <= 84:
        return ZONE_LETTERS[int(latitude + 80) >> 3]
    else:
        return None


def latlon_to_zone_number(latitude, longitude):
    # If the input is a numpy array, just use the first element
    # User responsibility to make sure that all points are in one zone
    if True:
        if isinstance(latitude, np.ndarray):
            latitude = latitude.flat[0]
        if isinstance(longitude, np.ndarray):
            longitude = longitude.flat[0]

    if 56 <= latitude < 64 and 3 <= longitude < 12:
        return 32

    if 72 <= latitude <= 84 and longitude >= 0:
        if longitude < 9:
            return 31
        elif longitude < 21:
            return 33
        elif longitude < 33:
            return 35
        elif longitude < 42:
            return 37

    return int((longitude + 180) / 6) + 1


def zone_number_to_central_longitude(zone_number):
    return (zone_number - 1) * 6 - 180 + 3
