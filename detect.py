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
input_size=YOLO_INPUT_SIZE

model = Create_Yolov3(input_size=input_size, CLASSES=TRAIN_CLASSES)
model.load_weights("./checkpoints/yolov3_custom")

dir_path = "Dataset\\Testing\\*"
print(len(glob.glob(dir_path)))
start = time.time()

for i,filename in enumerate(glob.glob(dir_path)):
    detect_image(model, filename, "Dataset/Testing_out/"+str(i+1)+".jpg", input_size=input_size, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
    print(i)
print("Time taken to run : "+str(time.time()-start))
