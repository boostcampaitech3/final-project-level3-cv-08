"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

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
        if abs(self.y[2][0]) > np.pi:
            if self.y[2][0] > 0 :
                self.y[2][0] = abs(self.y[2][0]) - 2*np.pi
            else:
                self.y[2][0] = 2*np.pi - abs(self.y[2][0])

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
def centerPointBatch(bb_test, bb_gt):
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
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    center_dist = ((bb_test[..., 0]-bb_gt[..., 0])**2 + (bb_test[..., 1]-bb_gt[..., 1])**2)**0.5
           
    return(center_dist)  

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x, y, rot, w, l, score] and returns z in the form
    [x,y,rot,w, l] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  new_bbox = bbox[:5] 
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
        self.cls_id = bbox[6]
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
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

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
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)[0], self.cls_id

    

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

    def update(self, dets=np.empty((0, 6))):
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
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.centerpoint_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d, cls_id = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1], [cls_id])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,7))

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

def Cam2LidarBev(calib, label):
    classes = ['Car', 'Pedestrian', 'Van', 'Cyclist']

    cls_id = label['cls_id']
    rect = calib['R0_rect']
    Tr_velo_to_cam = calib['Tr_velo_to_cam']
    rt_mat = np.linalg.inv(rect@Tr_velo_to_cam)
    # cam -> lidar

    frame = label['frame']

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

    def filterIdx(classes, label):
        idx = []
        for i in range(label['name'].shape[0]):
            if label['name'][i] in classes:
                idx.append(i)
        return idx
    
    filter_idx = filterIdx(classes, label)
    labels = label['name'][filter_idx]
    cls_id = label['cls_id'][filter_idx]
    tracking_id = label['tracking_id'][filter_idx]
    
    hwl_lidar = hwl_lidar[filter_idx]
    xyz_lidar = xyz_lidar[filter_idx]
    rotation_y_lidar = rotation_y_lidar[filter_idx]
    frame = frame[filter_idx]
    score_temp = np.ones_like(rotation_y_lidar) * 0.7

    detections = np.concatenate([frame[:, np.newaxis], xyz_lidar[:, :2], rotation_y_lidar[:,np.newaxis], hwl_lidar[:, 1:3], score_temp[:, np.newaxis], cls_id[:, np.newaxis], tracking_id[:, np.newaxis]], axis=1)
    return detections

def lidar2Bev(velodyne_path):
    vd = 0.4
    vh = 0.1
    vw = 0.1

    xrange = (0, 40.4)
    yrange = (-30, 30)
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
        return np.zeros((height, width, channel)), np.zeros((height, width))

    lidar = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)

    def filter_points(lidar, xrange, yrange, zrange):
        pxs = lidar[:, 0]
        pys = lidar[:, 1]
        pzs = lidar[:, 2]

        filter_x = np.where((pxs >= xrange[0]) & (pxs < xrange[1]))[0]
        filter_y = np.where((pys >= yrange[0]) & (pys < yrange[1]))[0]
        filter_z = np.where((pzs >= zrange[0]) & (pzs < zrange[1]))[0]
        filter_xy = np.intersect1d(filter_x, filter_y)
        filter_xyz = np.intersect1d(filter_xy, filter_z)

        return lidar[filter_xyz]

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

    for i in range(len(pxs)):
        top[-qxs[i], -qys[i], -1]= 1+ top[-qxs[i], -qys[i], -1]
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

    top[:,:,-1] = np.log(top[:,:,-1]+1)/math.log(64)
    # 이건 normalize방식인데 논문에 나와있음
    # top의 마지막 채널에는 point가 많을 경우 큰값, 적을 경우 작은값, 없을 경우 0인데, 이 값을 normalize함
    # log를 취하니깐 log1=0이니깐 값이 없으면 0, 아니면 logn/log64
    # 이렇게 되면 점이 많은 곳과 적은 곳의 색 차이가 나타나게 됨
    if 1:
        # top_image = np.sum(top[:,:,:-1],axis=2)
        density_image = top[:,:,-1]
        density_image = density_image-np.min(density_image)
        # 음수를 양수로 바꿔주기 위해 최소 값을 뺌
        density_image = (density_image/np.max(density_image)*255).astype(np.uint8)
        # 최대값으로 나눠주어 0~1사이 값으로 만들고 255를 곱해주어 픽셀 값으로 만들어줌
        # top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)


    return top, density_image

def bevPoints(trackers):
    xy = trackers[:, :2]
    rotation_y_lidar = trackers[:, 2]
    wl = trackers[:, 3:5]

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
        
        l = wl[i][1]/2
        w = wl[i][0]/2
        x_corners = [l, l, -l, -l];
        y_corners = [w, -w, -w, w];
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

def drawBbox(box3d, trackers, density_image, frame):
    classes = {-1:'DontCare', 0 : 'Car' , 1 : 'Pedestrian', 2 : 'Van', 3 : 'Cyclist'}
    ids = trackers[:, 5].reshape(-1).astype(np.int32)
    labels = trackers[:, 6].reshape(-1)
    img_2d = density_image[np.newaxis, :, :].repeat(3, 0).transpose(1, 2, 0)
    for i in range(box3d.shape[0]):
        color = compute_color_for_id(ids[i])
        points = box3d[i].reshape(-1, 2)
        points = points[:, ::-1]
        x_max = points[:,0].max()
        y_max = points[:,1].max()
        img_2d = cv2.polylines(img_2d, [points], True, color, thickness=2)
        cv2.putText(img_2d, classes[labels[i]]+str(ids[i]), (x_max, y_max), 0, 0.7, color, thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(img_2d, 'frame : ' + str(frame), (5, 30), 0, 0.7, color, thickness=1, lineType=cv2.LINE_AA)
    return img_2d

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='/content/gdrive/MyDrive/kitti_dataset')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='training')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--folder_id", type=str,default='0000')
    parser.add_argument("--centerpoint_threshold", help="Minimum centerpoint for match.", type=float, default=3.0)
    parser.add_argument("--save_path", type=str, default='/content/gdrive/MyDrive/sort_tracking')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
  # all train
  args = parse_args()
  display = args.display
  phase = args.phase
  total_time = 0.0
  total_frames = 0
  colours = np.random.randint(0, 256, (32, 3)) #used only for display

#   if(display):
#     if not os.path.exists('mot_benchmark'):
#       print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
#       exit()
#     plt.ion()
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111, aspect='equal')

  
  
  #pattern = os.path.join(args.seq_path, phase, 'KITTI-13', 'det', 'det_new.txt')
  label_path = os.path.join(args.seq_path, phase, 'label_02', '*.txt')
  # args.seq_path = base_path
  
  for seq_dets_fn in glob.glob(label_path):
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       centerpoint_threshold=args.centerpoint_threshold) #create instance of the SORT tracker
    seq_dets = load_kitti_label(seq_dets_fn)
    seq = seq_dets_fn.split('/')[-1].split('.')[0]
    # 0000, 0001, 0002 ....
    calib = load_kitti_calib(os.path.join(args.seq_path, phase, 'calib', '%s.txt'%seq))
    
    save_path = os.path.join(args.save_path, 'tracking_images', seq)
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    
    with open(os.path.join(save_path, '_%s.txt'%(seq)),'w') as out_file:
      print("Processing %s."%(seq))
      total_dets = Cam2LidarBev(calib, seq_dets)
      np.savetxt(os.path.join(save_path, '_gt_%s.txt'%(seq)), total_dets, fmt='%.2f',delimiter = ',')
      for frame in tqdm(range(int(total_dets[:,0].max()))):
        frame += 1 #detection and frame numbers begin at 1
        dets = total_dets[total_dets[:,0]==frame, 1:]
        total_frames += 1

        if(display):
          velodyne_path = os.path.join(args.seq_path, phase, 'velodyne', seq, '%06d.bin'%(frame))
          top, density_image = lidar2Bev(velodyne_path)
          
          #ax1.imshow(im)
          #plt.title(seq + ' Tracked Targets')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[5],d[0],d[1],d[2],d[3],d[4]),file=out_file)
          
        if(display):
            rotated_points = bevPoints(trackers)
            density_image = drawBbox(rotated_points, trackers, density_image, frame)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(os.path.join(save_path, '%06d.png'%(frame)), density_image)
       

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")
