"""
 @leofansq
 Main function
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import time
from tqdm import tqdm
from sklearn import linear_model
from func import find_files, cal_proj_matrix, cal_proj_matrix_raw, load_img, load_lidar, project_lidar2img

#**********************************************************#
#                         Option                           #
#**********************************************************#
################## FILE PATH ###################
# Calib File
CALIB_TYPE = 0      # 0:All parameters in one file. e.g. KITTI    1: Seperate into two files. e.g. KITTI raw
# if CALIB_TYPE == 0
CALIB = "./calib/000000.txt"
# if CALIB_TYPE == 1    
CAM2CAM = "./calib/calib_cam_to_cam.txt"
LIDAR2CAM = "./calib/calib_velo_to_cam.txt"

# Source File
IMG_PATH = "/opt/ml/Tools_Merge_Image_PointCloud-1/lane_seg_image/test_image/"
LIDAR_PATH = "/opt/ml/Tools_Merge_Image_PointCloud-1/lane_seg_image/test_velo/"

# Save File
#SIMG_PATH = "./result3/img/"
SBEV_PATH = "./result3/bev/"

################# PARAMETER ####################
CAM_ID = 2



def add_square_feature(X):
    X = np.concatenate([(X**2).reshape(-1,1), X], axis=1)
    return X

lane_t_num = 0
#**********************************************************#
#                     Main Function                        #
#**********************************************************#
def main():
    global lane_xy
    time_cost = []
    
    # Calculate P_matrix
    if CALIB_TYPE:
        p_matrix = cal_proj_matrix_raw(CAM2CAM, LIDAR2CAM, CAM_ID)
    else:
        p_matrix = cal_proj_matrix(CALIB, CAM_ID)

    # Batch Process
    for img_path in tqdm(find_files(IMG_PATH, '*.png')):
        
        _, img_name = os.path.split(img_path)
        pc_path = LIDAR_PATH + img_name[:-4] + '.bin'
        start_time = time.time()
        # Load img & pc
        img = load_img(img_path)
        pc = load_lidar(pc_path)
        
        # Project & Generate Image & Save
        points = project_lidar2img(img, pc, p_matrix)
        
        
        
        lane_coord = []
        pc = pc[(points[:,0]<1242) & (1<points[:,0]) & (1<points[:,1]) & (points[:,1]<375)]
        points = points[(points[:,0]<1242) & (1<points[:,0]) & (1<points[:,1]) & (points[:,1]<375)]

        LANE_IMG = cv2.imread(IMG_PATH + img_name)
        LANE_IMG = LANE_IMG[:,:,0]
        LANE_IMG = np.array(LANE_IMG, dtype=np.float32)
        LANE_IMG = cv2.resize(LANE_IMG, (1242, 375))

        for idx,i in enumerate(points):
            if LANE_IMG[int(i[1]),int(i[0])]:
                lane_coord.append(idx)
                


        img_bev = np.zeros((800, 700, 3))
        
        lane_xy = []
        for i in lane_coord:
            lane_xy.append(list(pc[i,:2]))
        lane_xy.sort()
        lane_xy = np.array(lane_xy)
        
        
        global lane_0
        lane_0 = lane_xy[0:1]
        def lane_classification(idx, lane_number, max_lane):
            global lane_t_num
            if abs(lane_xy[idx,1] - lane_xy[idx+1,1]) <= 1:
                if len(lane_xy) > idx+2:
                    globals()[f'lane_{lane_number}'] = np.vstack((globals()[f'lane_{lane_number}'], lane_xy[idx+1:idx+2]))
                    lane_classification(idx+1, lane_number, max_lane)

            elif len(lane_xy) > idx+2:
                
                s = 1
                for i in range(max_lane+1):

                    if abs(globals()[f'lane_{i}'][-1, 1] - lane_xy[idx+1, 1]) <= 1:

                        globals()[f'lane_{i}'] = np.vstack((globals()[f'lane_{i}'], lane_xy[idx+1:idx+2]))
                        s = 0
                        lane_number = i
                        break
                if s:
                    max_lane += 1
                    globals()[f'lane_{max_lane}'] = lane_xy[idx+1:idx+2]
                    lane_number = max_lane
                    lane_t_num = max_lane + 1
                lane_classification(idx+1, lane_number, max_lane)

        
        

        lane_classification(0, 0, 0)
        LANE = []
        LANE_DICT = {}
        
        for i in range(lane_t_num):
            if len(globals()[f'lane_{i}']) >= len(lane_xy)//20:
                LANE.append(globals()[f'lane_{i}'])
        for i in LANE:
            LANE_DICT[abs(np.mean(i[:,1]))] = i
        LANE_OURS = [LANE_DICT.pop(min(LANE_DICT)), LANE_DICT.pop(min(LANE_DICT))]
        
      

        for lane_xy in LANE_OURS:
            X = lane_xy[:,0]
            y = lane_xy[:,1]
            X = X.reshape(-1, 1)

            # Fit line using all data
            lr = linear_model.LinearRegression()
            lr.fit(add_square_feature(X), y)

            # Robustly fit linear model with RANSAC algorithm
            ransac = linear_model.RANSACRegressor()
            ransac.fit(add_square_feature(X), y)
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)

            # Predict data of estimated models
            line_X = np.arange(X.min(), X.max())[:, np.newaxis]
            line_y_ransac = ransac.predict(add_square_feature(line_X))

            for i in range(len(line_X)-1):
                cv2.line(img_bev, (int(-line_y_ransac[i]*10)+350, -int(line_X[i]*10)+799), (int(-line_y_ransac[i+1]*10)+350, -int(line_X[i+1]*10)+799), (0,255,0), 3)

        cv2.imwrite(SBEV_PATH+img_name[:-4]+"_bev.png", img_bev)

        
        end_time = time.time()
        time_cost.append(end_time - start_time)

    print ("Mean_time_cost:", np.mean(time_cost))
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()