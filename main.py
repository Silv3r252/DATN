import cv2
import os
from yolov9.my_detect import run
import sys
sys.path.append("E:\WorkSpace\Python\Do_An_Tot_Nghiep\yolov9\my_WPOD")
sys.path.append("E:\WorkSpace\Python\Do_An_Tot_Nghiep\yolov9\my_WPOD\src")
from yolov9.my_WPOD import lp_detection
from yolov9.my_WPOD import src


if __name__ == '__main__':
    # data_img = os.path.join("Data","img","img_01.jpg")
    data_img = os.path.join("Data", "img", "frame_10.jpg")
    img_path = "E:/ThucTeAo/BTL_XuLyAnh/Data_gia/x/frame_92.jpg"
    data_vid = os.path.join("Data","video","test_2.mp4")
    model_path = os.path.join("models","best.pt")
    conf_thres = 0.6
    run(weights=model_path,source=data_vid,conf_thres=conf_thres,view_img=True,nosave=True)
