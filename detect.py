#================================================================
#
#   File name   : detect_mnist.py
#   Author      : PyLessons
#   Created date: 2020-04-20
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : mnist object detection example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
import numpy as np
import random
import time
import tensorflow as tf
from yolov3.yolov3 import Create_Yolov3
from yolov3.utils import detect_image
from yolov3.configs import *

# from tensorflow.contrib.lite.python import lite
from tensorflow.keras import Input, Model

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
input_size=YOLO_INPUT_SIZE

ID = random.randint(0, 200)
# label_txt = "mnist/mnist_test.txt"
# image_info = open(label_txt).readlines()[ID].split()

# image_path = image_info[0]

model = Create_Yolov3(input_size=input_size, CLASSES=TRAIN_CLASSES)
model.load_weights("./checkpoints/yolov3_custom") # use keras weights

model_path = os.path.join(ROOT_DIR, 'model', 'yolov3.h5')

tf.keras.models.save_model(model, model_path, overwrite=True)

# Sanity check to see if model loads properly
# NOTE: See https://github.com/keras-team/keras/issues/4609#issuecomment-329292173
# on why we have to pass in `tf: tf` in `custom_objects`
model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf})
# input image
IMAGE_HEIGHT = 416
IMAGE_WIDTH = 416
CHANNELS = 3
model._set_inputs(inputs=tf.random.normal(shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter = tf.lite.TocoConverter.from_keras_model_file(model_path,input_shapes={'input_1': [1, config['width'], config['height'], 3]})
converter.post_training_quantize = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

open("model/yolov3.tflite", "wb").write(tflite_model)
# model_json = yolo.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# yolo.save_weights("model.h5")
# print("Saved model to disk")
# for i in range(6):
#     image_path = "Dataset/Test2/"+str(i+1)+".jpg"
#     detect_image(model, image_path, "", input_size=input_size, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#     time.sleep(10)

