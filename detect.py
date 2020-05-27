import os
import cv2
import numpy as np
import random
import time
import tensorflow as tf
from yolov3.yolov3 import Create_Yolov3
from yolov3.utils import detect_image
from yolov3.configs import *
import glob
import time

from tensorflow.keras import Input, Model

IMAGE_HEIGHT = 416
IMAGE_WIDTH = 416
CHANNELS = 3

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
io=YOLO_INPUT_SIZE

yolo_model = Create_Yolov3(input_size=io, CLASSES=TRAIN_CLASSES)
yolo_model.load_weights("./checkpoints/yolov3_custom")

dir_path = "Dataset\\Testing\\*"

st_time = time.time()

for file_no,file_img in enumerate(glob.glob(dir_path)):
    detect_image(yolo_model, file_img, "Dataset/Testing_out/"+str(file_no+1)+".jpg", input_size=io, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
	
finish_time=time.time()

print("Total Time: "+str(finish_time-st_time))