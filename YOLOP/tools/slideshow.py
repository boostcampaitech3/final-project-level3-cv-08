import cv2
import glob

cv2.namedWindow('demo', cv2.WINDOW_AUTOSIZE)

files_path = ""

img_frames = glob.glob(files_path + "/*")

for frame in img_frames:
    img = cv2.imread(frame)
    cv2.imshow('demo', img)
    
    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()

