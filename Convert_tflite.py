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
# from tensorflow.contrib.lite.python import lite
from tensorflow.keras import Input, Model

IMAGE_HEIGHT = 416
IMAGE_WIDTH = 416
CHANNELS = 3
input_size=YOLO_INPUT_SIZE

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

model = Create_Yolov3(input_size=input_size, CLASSES=TRAIN_CLASSES)
model.load_weights("./checkpoints/yolov3_custom") # use keras weights

model_path = os.path.join(ROOT_DIR, 'model', 'yolov3.h5')
tf.keras.models.save_model(model, model_path, overwrite=True)

model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf})
model._set_inputs(inputs=tf.random.normal(shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.post_training_quantize = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open("model/yolov3.tflite", "wb").write(tflite_model)
print("Saved model")