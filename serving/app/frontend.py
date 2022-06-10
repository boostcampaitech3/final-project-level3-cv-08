import io
import os
from pathlib import Path

import requests
from PIL import Image

import streamlit as st
from app.confirm_button_hack import cache_on_button_press
import numpy as np
from urllib import request
import ffmpeg
import cv2
import time
import tempfile

import urllib.request

# SETTING PAGE CONFIG TO WIDE MODE
ASSETS_DIR_PATH = os.path.join(Path(__file__).parent.parent.parent.parent, "assets")

st.set_page_config(layout="wide")

root_password = 'password'

def openImg(url):
    res = request.urlopen(url).read()
    img = Image.open(io.BytesIO(res))
    return img

def openVid(url, vid_name):
    request.urlretrieve(url, vid_name)
    vid = open(vid_name, 'rb')
    return vid

scene_a = openImg("https://storage.googleapis.com/pre-saved/Image/Image_A.jpg")
scene_b = openImg('https://storage.googleapis.com/pre-saved/Image/Image_B.jpg')
scene_c = openImg('https://storage.googleapis.com/pre-saved/Image/Image_C.jpg')

video_a = openVid("https://storage.googleapis.com/pre-saved/Video/Video_1.mp4", "data/Video_A.mp4")
video_b = openVid("https://storage.googleapis.com/pre-saved/Video/Video_B.mp4", "data/Video_B.mp4")
result_video_b = openVid("https://storage.googleapis.com/pre-saved/Video/LANE_result_B.mp4", "data/result_Video_B.mp4")
video_c = openVid("https://storage.googleapis.com/pre-saved/Video/Video_C.mp4", "data/Video_C.mp4")
result_video_c = openVid("https://storage.googleapis.com/pre-saved/Video/LANE_result_C.mp4", "data/result_Video_C.mp4")

fusion_scenes_a = openImg('https://storage.googleapis.com/pre-saved/Image/Image_C.jpg')
fusion_scenes_b = openImg('https://storage.googleapis.com/pre-saved/Image/Image_C.jpg')
fusion_scenes_c = openImg('https://storage.googleapis.com/pre-saved/Image/Image_C.jpg')

fusion_result_a = openImg('https://storage.googleapis.com/pre-saved/Image/Image_C.jpg')
fusion_result_b = openImg('https://storage.googleapis.com/pre-saved/Image/Image_C.jpg')
fusion_result_c = openImg('https://storage.googleapis.com/pre-saved/Image/Image_C.jpg')

fusion_video_a = openVid("https://storage.googleapis.com/pre-saved/Video/Video_C.mp4", "data/Video_C.mp4")
fusion_video_b = openVid("https://storage.googleapis.com/pre-saved/Video/Video_C.mp4", "data/Video_C.mp4")
fusion_video_c = openVid("https://storage.googleapis.com/pre-saved/Video/Video_C.mp4", "data/Video_C.mp4")

Scenes = {'Scene A': scene_a, 'Scene B': scene_b, 'Scene C': scene_c}
Fusion_Scenes = {'Scene A': fusion_scenes_a, 'Scene B': fusion_scenes_b, 'Scene C': fusion_scenes_c}
Fusion_Result = {'Scene A': fusion_result_a, 'Scene B': fusion_result_b, 'Scene C': fusion_result_c}
Videos = {'Video A': video_a, 'Video B': video_b, 'Video C': video_c}
Fusion_Video_Result = {'Video A': fusion_video_a, 'Video B': fusion_video_b, 'Video C': fusion_video_c}

inf_time_b = 0.0
inf_time_c = 0.0
result_Videos = {'Video B': (result_video_b, inf_time_b), 'Video C': (result_video_c, inf_time_c)}



def main():
    files = []

    st.title("Autonomous Mad Max")
    #st.title("CV08조 주행청소년 최종프로젝트")
    st.caption("")
    st.caption("")
    
    col1, col3, col4 = st.columns(3)

    with col1:
        selected_item = st.radio('Camera or Fusion(Camera & Lidar)', ('Camera', 'Fusion(Camera & Lidar)'))
    if selected_item == 'Fusion(Camera & Lidar)':
        st.error("The 3D Model is preparing...")
    with col3:
        if selected_item == "Camera":
            image_or_video = st.radio('Image or Video', ('Image', 'Video'))
        else:
            image_or_video = st.radio('Image or Video', ('Video',))

    if image_or_video == 'Image':
        with col4:
            option = st.radio('Please choose scene or upload image', ('Scene A', 'Scene B', 'Scene C', 'Upload'))

        if option == 'Upload':
            if selected_item == 'Camera':
                uploaded_img = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
                if uploaded_img:
                    image_bytes = uploaded_img.getvalue()
                    files = [
                        ('files', (uploaded_img.name, image_bytes,
                                uploaded_img.type))
                    ]

            elif selected_item == 'Fusion(Camera & Lidar)':
                uploaded_img = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
                uploaded_lidar = st.file_uploader("Choose an lidar", type=["bin", "bcd.bin", "bcd"])
                if uploaded_img and uploaded_lidar:
                    image_bytes = uploaded_img.getvalue()
                    lidar_bytes = uploaded_lidar.getvalue()
                    files = [
                        ('files', (uploaded_img.name, image_bytes,
                                    uploaded_img.type)), 
                        ('files', (uploaded_lidar.name, lidar_bytes,
                                    uploaded_lidar.type))
                    ]
        else:
            st.text('')
            if selected_item == 'Camera':
                st.image(Scenes[option], caption=f'{option}')
            elif selected_item == 'Fusion(Camera & Lidar)':
                st.image(Fusion_Scenes[option], caption=f'{option}')

        def print_both_scene_result(response):
            st.text(f'Lidar Inference Time: {response.json()["lidar_inf_time"]:.5f}s')
            st.text(f'Lidar Plot Time: {response.json()["lidar_plot_time"]:.5f}s')
            print_image_scene_result(response)

        def print_image_scene_result(response):
            st.text('Inference Time은 한 Frame당 약 0.03s정도 이지만 서빙 과정에서의 문제로 훨씬 길어질 수 있습니다.')
            st.text(f'Image Inference Time: {response.json()["image_inf_time"]:.5f}s')
            st.text(f'Image Plot Time: {response.json()["image_plot_time"]:.5f}s')
            with st.spinner(text='Image Loading...'):
                img = Image.fromarray(np.array(response.json()["products"][0]["result"][0]).astype('uint8'))
                st.image(img, caption="Image Demo")

        if st.button('Inference'):
            if selected_item == 'Fusion(Camera & Lidar)':
                if option != 'Upload':
                    st.image(Fusion_Result[option])
                    #response = requests.post(f"http://localhost:8001/both_prepared_order", option=option)
                    #print_both_scene_result(response)
                else:
                    with st.spinner(text='In progress...'):
                        if files:
                            st.error("The 3D Model is preparing...")
                            #response = requests.post(f"http://localhost:8001/both_order", files=files)
                            #print_both_scene_result(response)
                        else:
                            st.error("Please upload files!!")

            elif selected_item == 'Camera':
                if option != 'Upload':
                    with st.spinner(text='In progress...'):
                        response = requests.post(f"http://localhost:8001/prepared_order/{option}")
                        print_image_scene_result(response)
                else:
                    with st.spinner(text='In progress...'):
                        if files:
                            response = requests.post(f"http://localhost:8001/order", files=files)
                            print_image_scene_result(response)
                        else:
                            st.error("Please upload image!!")

    elif image_or_video == 'Video':
        with col4:
            if selected_item == "Camera":
                option = st.radio('Please choose or upload video', ('Video A', 'Video B', 'Video C', 'Upload'))
            else:
                option = st.radio('Please choose or upload video', ('Video A', 'Video B', 'Video C'))
        st.info('Video A는 직접 모델을 거쳐서 시연되고 Video B, C는 결과물만 보여줍니다.')
        if option == 'Upload':
            st.info("300 frame 이하의 영상만 inference 가능합니다.")
            if selected_item == 'Camera':
                uploaded_video = st.file_uploader("Upload an video", type=["mp4", "mov", "avi"])
                if uploaded_video:
                    video_bytes = uploaded_video.getvalue()
                    files = [
                        ('files', (uploaded_video.name, video_bytes,
                                uploaded_video.type))
                    ]

            elif selected_item == 'Fusion(Camera & Lidar)':
                uploaded_video = st.file_uploader("Upload an video", type=["mp4", "mov", "avi"])
                uploaded_lidar = st.file_uploader("Upload an lidar", type=["bin", "bcd.bin", "bcd"], accept_multiple_files=True)
        
        else:
            video_bytes = Videos[option].read()
            st.video(video_bytes)
            
        
        def print_both_video_result(response):
            st.text(f'Lidar Inference Time: {response.json()["lidar_inf_time"]:.5f}s')
            st.text(f'Lidar Plot Time: {response.json()["lidar_plot_time"]:.5f}s')
            print_image_video_result(response)

        def print_image_video_result(response):
            st.text('GPU가 모델학습에 사용중이라면 Inference Time이 0.03s 보다 훨씬 길어질 수 있습니다.')
            st.text(f'Image Inference Time: {response.json()["image_inf_time"]:.5f}s')
            with st.spinner(text='Image Loading...'):
                video_path = response.json()["products"][0]["result"][0]
                video_bytes = open(video_path, 'rb').read()
                st.video(video_bytes)


        if st.button('Inference'):
            if selected_item == 'Fusion(Camera & Lidar)':
                if option != 'Upload':
                    st.video(Fusion_Video_Result[option])
                    #response = requests.post(f"http://localhost:8001/both_prepared_order", option=option)
                    #print_both_video_result(response)
                else:
                    st.text("remove the fusion's file uploader!!!!!")

            elif selected_item == 'Camera':
                if option != 'Upload':
                    with st.spinner(text="Inference 진행중... 최대 3분 30초 정도 걸릴 수 있습니다"):
                        if option == 'Video A':
                            response = requests.post(f"http://localhost:8001/prepared_order/{option}")
                            print_image_video_result(response)
                        else:
                            st.text('GPU가 모델학습에 사용중이라면 Inference Time이 0.03s 보다 훨씬 길어질 수 있습니다.')
                            st.text(f'Inference Time Per Frame: {result_Videos[option][1]:.3f}s')
                            with st.spinner(text='Video Loading...'):
                                video_bytes = result_Videos[option][0]
                                st.video(video_bytes)
                            
                else:
                    with st.spinner(text="Inference 진행중... 최대 5분 정도 걸릴 수 있습니다"):
                        if files:
                            #st.error("Preparing...")
                            #########################################################################
                            import tempfile

                            tfile = tempfile.NamedTemporaryFile(delete=False)
                            tfile.write(uploaded_video.read())
                            #stframe = st.empty()
                            vf = cv2.VideoCapture(tfile.name)
                            height, width, _ = vf.read()[1].shape
                            fps = vf.get(cv2.CAP_PROP_FPS)
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            out = cv2.VideoWriter('data/uploaded.mp4', fourcc, fps, (width, height), True)
                            stframe = st.empty()
                            i = 0
                            while vf.isOpened():
                                ret, frame = vf.read()
                                stframe.write(f"{i}번째 frame 저장중....")
                                if not ret:
                                    print("Can't receive frame (stream end?). Exiting ...")
                                    break
                                out.write(frame)
                                i += 1
                            out.release()
                            if i >= 300:
                                st.error("300 프레임 이상의 영상은 inference 불가능합니다!")
                            else:
                                stframe.write("영상 저장 완료!")
                                response = requests.post(f"http://localhost:8001/prepared_order/{option}")
                                print_image_video_result(response)
                        else:
                            st.error("Please upload image!!")





@cache_on_button_press('Authenticate')
def authenticate(password) -> bool:
    return password == root_password


#password = st.text_input('password', type="password")
#if authenticate(password):
#    st.success('You are authenticated!')
#else:
#    st.error('The password is invalid.')

main()
