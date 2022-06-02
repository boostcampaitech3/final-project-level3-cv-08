# -*- coding: utf-8 -*-
import cv2

def imageCopy(image):
    img_src = cv2.imread(image)
    img_clone = img_src.copy()
    return img_clone

#라인 그리기
def drawLine(image, point1, point2, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA):
    result = imageCopy(image)
    return cv2.line(result, point1, point2, color, thickness, lineType)

#원 그리기
def drawCircle(image, center, radius, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA):
    result = imageCopy(image)
    return cv2.circle(result, center, radius, color, thickness, lineType)

#사각형 그리기
def drawRect(image, point1, point2, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA):
    result = imageCopy(image)
    return cv2.rectangle(result, point1, point2, color, thickness, lineType)

def imageProcessing2(image):
    result = imageCopy(image)
    pt1 = (430, 310)
    pt2 = (530, 310)
    pt3 = (940, 540)
    pt4 = (20, 540)
    result = drawLine(result, pt1, pt2, (255, 0, 0), 5)
    result = drawLine(result, pt2, pt3, (255, 0, 0), 5)
    result = drawLine(result, pt3, pt4, (255, 0, 0), 5)
    result = drawLine(result, pt4, pt1, (255, 0, 0), 5)
    height = image.shape[0]
    width = image.shape[1]
    pt1 = (int(width * 0.5), int(height * 0.5))
    pt2 = (int(width), int(height))
    pt3 = (0, height)
    result = drawLine(result, pt1, pt2, (0, 255, 0), 5)
    result = drawLine(result, pt2, pt3, (0, 255, 0), 5)
    result = drawLine(result, pt3, pt1, (0, 255, 0), 5)
    pt1 = (0,0)
    pt2 = (width, height)
    result = drawRect(result, pt1, pt2, (0, 0, 0), 5)
    pt1 = (int(width * 0.5), int(height * 0.5))
    result = drawCircle(result, pt1, 3, (255, 255, 255), -1)
    return result

def Video2(openpath, savepath = "output.mp4"):
    cap = cv2.VideoCapture(openpath)
    if cap.isOpened():
        print("Video Opened")
    else:
        print("Video Not Opened")
        print("Program Abort")
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(savepath, fourcc, fps, (width, height), True)
    cv2.namedWindow("Input", cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            output = imageProcessing2(frame)
            out.write(output)
            cv2.imshow("Input", frame)
            cv2.imshow("Output", output)
        else:
            break
        if cv2.waitKey(int(1000.0/fps)) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return

road_video_01 = "./clips/PADUA/clip_000000.mp4"

Video2(road_video_01, "output.mp4")