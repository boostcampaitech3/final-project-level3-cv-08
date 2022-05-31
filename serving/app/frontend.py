import io
import os
from pathlib import Path

import requests
from PIL import Image

import streamlit as st
from app.confirm_button_hack import cache_on_button_press
import numpy as np

# SETTING PAGE CONFIG TO WIDE MODE
ASSETS_DIR_PATH = os.path.join(Path(__file__).parent.parent.parent.parent, "assets")

st.set_page_config(layout="wide")

root_password = 'password'
scene_a = Image.open('/opt/ml/bdd_for_yolop/bdd100k/images/100k/test/fcd22a1c-d019a362.jpg')
scene_b = Image.open('/opt/ml/bdd_for_yolop/bdd100k/images/100k/test/cabf7be1-36a39a28.jpg')
scene_c = Image.open('/opt/ml/bdd_for_yolop/bdd100k/images/100k/test/f3c744e5-1f611c0a.jpg')

video_a = open('/opt/ml/final-project-level3-cv-08/YOLOP/inference/videos/1.mp4', 'rb')
video_b = open('/opt/ml/final-project-level3-cv-08/YOLOP/inference/videos/1.mp4', 'rb')
video_c = open('/opt/ml/final-project-level3-cv-08/YOLOP/inference/videos/1.mp4', 'rb')


Scenes = {'Scene A': scene_a, 'Scene B': scene_b, 'Scene C': scene_c}

Videos = {'Video A': video_a, 'Video B': video_b, 'Video C': video_c}



def main():
    files = []

    st.title("Autonomous Mad Max")
    #st.title("CV08조 주행청소년 최종프로젝트")
    st.caption("")
    st.caption("")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        selected_item = st.radio('Camera or Fusion(Camera & Lidar)', ('Camera', 'Fusion(Camera & Lidar)'))
    if selected_item == 'Fusion(Camera & Lidar)':
        st.error("The 3D Model is preparing...")
    with col2:
        region = st.radio('Region', ('USA', 'Korea'))
    with col3:
        image_or_video = st.radio('Image or Video', ('Image', 'Video'))

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
                st.error("The 3D Model is preparing...")
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
            st.image(Scenes[option], caption=f'{option}')

        def print_both_scene_result(response):
            st.text(f'Lidar Inference Time: {response.json()["lidar_inf_time"]:.5f}s')
            st.text(f'Lidar Plot Time: {response.json()["lidar_plot_time"]:.5f}s')
            print_image_scene_result(response)

        def print_image_scene_result(response):
            st.text('GPU가 모델학습에 사용중이라면 Inference Time이 0.03s 보다 훨씬 길어질 수 있습니다.')
            st.text(f'Image Inference Time: {response.json()["image_inf_time"]:.5f}s')
            st.text(f'Image Plot Time: {response.json()["image_plot_time"]:.5f}s')
            with st.spinner(text='Image Loading...'):
                img = Image.fromarray(np.array(response.json()["products"][0]["result"][0]).astype('uint8'))
                st.image(img, caption="Image Demo")

        if st.button('Inference'):
            if selected_item == 'Fusion(Camera & Lidar)':
                if option != 'Upload':
                    with st.spinner(text='In progress...'):
                        response = requests.post(f"http://localhost:8001/both_prepared_order", option=option)
                        print_both_scene_result(response)
                else:
                    st.error("The 3D Model is preparing...")
                    with st.spinner(text='In progress...'):
                        if files:
                            response = requests.post(f"http://localhost:8001/both_order", files=files)
                            print_both_scene_result(response)
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
        st.error('Preparing...')
        with col4:
            option = st.radio('Please choose or upload video', ('Video A', 'Video B', 'Video C', 'Upload'))
        
        if option == 'Upload':
            if selected_item == 'Camera':
                uploaded_video = st.file_uploader("Upload an video", type=["mp4", "mov", "avi"])
                if uploaded_video:
                    video_bytes = uploaded_video.getvalue()
                    files = [
                        ('files', (uploaded_video.name, video_bytes,
                                uploaded_video.type))
                    ]

            elif selected_item == 'Fusion(Camera & Lidar)':
                st.error("The 3D Model is preparing...")
                uploaded_video = st.file_uploader("Upload an video", type=["mp4", "mov", "avi"])
                uploaded_lidar = st.file_uploader("Upload an lidar", type=["bin", "bcd.bin", "bcd"], accept_multiple_files=True)
        
        else:
            video_bytes = Videos[option].read()
            st.video(video_bytes)
        


        if st.button('Inference'):
            if selected_item == 'Fusion(Camera & Lidar)':
                if option != 'Upload':
                    with st.spinner(text='In progress...'):
                        response = requests.post(f"http://localhost:8001/both_prepared_order", option=option)
                        print_both_result(response)
                else:
                    st.error("The 3D Model is preparing...")
                    with st.spinner(text='In progress...'):
                        if files:
                            response = requests.post(f"http://localhost:8001/both_order", files=files)
                            print_both_result(response)
                        else:
                            st.error("Please upload files!!")

            elif selected_item == 'Camera':
                if option != 'Upload':
                    with st.spinner(text='In progress...'):
                        response = requests.post(f"http://localhost:8001/prepared_order/{option}")
                        print_image_result(response)
                else:
                    with st.spinner(text='In progress...'):
                        if files:
                            response = requests.post(f"http://localhost:8001/order", files=files)
                            print_image_result(response)
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
