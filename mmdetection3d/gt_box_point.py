from cmath import sin, cos
from uuid import NAMESPACE_URL
from mmdet3d.core.evaluation.kitti_utils.rotate_iou import rotate_iou_gpu_eval
import pickle
import numpy as np

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
        obj_class = []
        for i in range(label['name'].shape[0]):
            if label['name'][i] in classes and label['frame'][i]==frame:
                idx.append(i)
                obj_class.append(label['name'][i])
        return idx, obj_class

    
    
    filter_idx, obj_class = filterIdx(classes, label, frame)
    
    hwl_lidar = hwl_lidar[filter_idx]
    xyz_lidar = xyz_lidar[filter_idx]
    rotation_y_lidar = rotation_y_lidar[filter_idx]


    gt = np.concatenate([xyz_lidar[:,:3], hwl_lidar, rotation_y_lidar[:,np.newaxis]], axis=1)

    return gt, obj_class

def main():
    with open('/opt/ml/final-project-level3-cv-08/mmdetection3d/result.pkl', 'rb') as file:
        outputs = pickle.load(file)

    label = load_kitti_label('/opt/ml/final-project-level3-cv-08/mmdetection3d/z_point_analize/label0000.txt')
    calib = load_kitti_calib('/opt/ml/final-project-level3-cv-08/mmdetection3d/z_point_analize/0000.txt')

    f = open('gt_box_point_num.txt', 'w')

    for cur_frame in range(len(outputs)):
        gts, obj_cls = Cam2LidarBev(calib, label, cur_frame)

        # bin file forder path
        pcd = np.fromfile(f'/opt/ml/final-project-level3-cv-08/mmdetection3d/z_point_analize/0000/{str.zfill(str(cur_frame), 6)}.bin', dtype='float32').reshape(-1, 4)

        '''
        bbox_1p : gt_box 왼쪽 상단 꼭지점
        bbox_2p : gt_box 오른쪽 상단 꼭지점
        bbox_3p : gt_box 오른쪽 하단 꼭지점
        '''
        for i, gt in enumerate(gts):
            bbox_x = gt[0]
            bbox_y = gt[1]
            bbox_z = gt[2]
            bbox_h = gt[3]
            bbox_w = gt[4]
            bbox_l = gt[5]
            rotate = gt[6]
            point = 0
            bbox_1p = np.array([bbox_l/2, bbox_w/2])
            bbox_2p = np.array([bbox_l/2, -bbox_w/2])
            bbox_3p = np.array([-bbox_l/2, -bbox_w/2])
            bbox_1p = np.array([bbox_1p[0]*cos(rotate) - bbox_1p[1]*sin(rotate) + bbox_x, bbox_1p[0]*sin(rotate) + bbox_1p[1]*cos(rotate) + bbox_y])
            bbox_2p = np.array([bbox_2p[0]*cos(rotate) - bbox_2p[1]*sin(rotate) + bbox_x, bbox_2p[0]*sin(rotate) + bbox_2p[1]*cos(rotate) + bbox_y])
            bbox_3p = np.array([bbox_3p[0]*cos(rotate) - bbox_3p[1]*sin(rotate) + bbox_x, bbox_3p[0]*sin(rotate) + bbox_3p[1]*cos(rotate) + bbox_y])
            a_vec = bbox_1p - bbox_2p
            b_vec = bbox_3p - bbox_2p

            for coordi in pcd:
                x = coordi[0]
                y = coordi[1]
                z = coordi[2]
                c_vec = np.array([x, y]) - bbox_2p
                if np.dot(a_vec, c_vec) >= 0 and np.dot(b_vec, c_vec) >= 0 and np.dot(a_vec, a_vec) >= np.dot(a_vec, c_vec) and np.dot(b_vec, b_vec) >= np.dot(b_vec, c_vec):
                    if bbox_z - bbox_h/2 <= z <= bbox_z + bbox_h/2:
                        point += 1
            f.write(f'cur_frame : {cur_frame}, class : {obj_cls[i]}, distance : {round((bbox_x**2 + bbox_y**2)**0.5, 2)}, point_num = {point}\n')
    f.close()
   

if __name__ == '__main__':
    main()