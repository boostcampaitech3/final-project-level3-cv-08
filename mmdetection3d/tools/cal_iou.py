from uuid import NAMESPACE_URL
from mmdet3d.core.evaluation.kitti_utils.rotate_iou import rotate_iou_gpu_eval
import pickle
import numpy as np
import cv2
import math
import os
#0 0 Van 0 0 -1.793451 296.744956 161.752147 455.226042 292.372804 2.000000 1.823255 4.433886 -4.552284 1.858523 13.410495 -2.115488
#frame tracking_id cls_name none none none x1 y1 x2 y2 x y z h w l rot
def load_kitti_label(label_file):
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

def load_kitti_calib(calib_file):
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
    print('finished')
    return {'P0' : P0, 'P1' : P1, 'P2' : P2, 'P3' : P3, 'R0_rect' : R0_rect,
            'Tr_velo_to_cam' : Tr_velo_to_cam, 'Tr_imu_to_velo' : Tr_imu_to_velo}

def Cam2LidarBev(calib, label, frame):
    classes = ['Car', 'Pedestrian', 'Van', 'Cyclist']

    rect = calib['R0_rect']
    Tr_velo_to_cam = calib['Tr_velo_to_cam']
    rt_mat = np.linalg.inv(rect@Tr_velo_to_cam)
    # cam -> lidar

    names = label['name']

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
    names = names[filter_idx]

    gt = np.concatenate([xyz_lidar[:, :2], hwl_lidar[:, 1:3], rotation_y_lidar[:,np.newaxis]], axis=1)
    return gt, names

def bevPoints(trackers):
    xy = trackers[:, :2]
    rotation_y_lidar = trackers[:, 4]
    lw = trackers[:, 2:4]

    v = 0.1
    xrange = (0, 70.4)
    yrange = (-50, 50)
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

def main():
    with open('/opt/ml/result.pkl', 'rb') as file:
        outputs = pickle.load(file)

    if not os.path.exists('/opt/ml/iou_images'):
        os.makedirs('/opt/ml/iou_images')

    
    label = load_kitti_label('/opt/ml/kitti1/label_0000.txt')
    calib = load_kitti_calib('/opt/ml/kitti1/testing/calib/calib_0000.txt')
    ious = []
    
    FN = []
    for cur_frame in range(len(outputs)):
        gt, name = Cam2LidarBev(calib, label, cur_frame)
        gt = np.array(gt)[:, [0, 1, 3, 2, 4]]
        # x, y, length, width rotation

        filter_idx = [0, 1, 3, 4, 6]
        detections = np.array(outputs[cur_frame]['boxes_3d'][:, filter_idx])
        # x, y, length, width rotation
        
        iou = rotate_iou_gpu_eval(gt, detections)
        
        ious.append(iou)
        

        img_2d = np.zeros((1000, 1000, 3))
        
        points = bevPoints(detections)
        gt_points = bevPoints(gt)

        for point in points:
            point = point.reshape(-1, 2).astype(np.int32)[:,::-1]
            img_2d = cv2.polylines(img_2d, [point], True, (255, 0, 0), thickness=2)

        for point in gt_points:
            point = point.reshape(-1, 2).astype(np.int32)[:,::-1]
            img_2d = cv2.polylines(img_2d, [point], True, (0, 0, 255), thickness=2)
        cv2.imwrite(f'/opt/ml/iou_images/img{cur_frame:06d}.png', img_2d)

    with open('/opt/ml/iou_images/iou_results.pkl', 'wb') as file:
        pickle.dump(ious, file)

        
        

if __name__ == '__main__':
    main()