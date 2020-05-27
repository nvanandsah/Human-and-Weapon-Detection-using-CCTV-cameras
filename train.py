import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import shutil
import numpy as np
import tensorflow as tf
#from tensorflow.keras.utils import plot_model
#from tqdm import tqdm
from yolov3.dataset import Dataset
from yolov3.yolov3 import Create_Yolov3, YOLOv3, decode, compute_loss
from yolov3.utils import load_yolo_weights
from yolov3.configs import *

input_size = YOLO_INPUT_SIZE
logdir = TRAIN_LOGDIR
Darknet_w = YOLO_DARKNET_WEIGHTS

save_best_only = True # saves only best agent according validation loss
save_checkpoints = False # saves all best validates checkpoints in training process (may require a lot disk space)

if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)

testing_dataset = Dataset('test')
training_dataset = Dataset('train')
steps = len(training_dataset)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
total_steps = TRAIN_EPOCHS * steps
warmup_steps = TRAIN_WARMUP_EPOCHS * steps

if TRAIN_TRANSFER:
    Darknet_obj = Create_Yolov3(input_size=input_size)
    load_yolo_weights(Darknet_obj, Darknet_w) 

yolo = Create_Yolov3(input_size=input_size, training=True, CLASSES=TRAIN_CLASSES)

if TRAIN_TRANSFER:
    for i, l in enumerate(Darknet_obj.layers):
        l_weights = l.get_weights()
        if l_weights == []:
            print("Weights not loaded")
        else:
            try:
                yolo.layers[i].set_weights(l_weights)
            except:
                print("No include", yolo.layers[i].name)

optimizer = tf.keras.optimizers.Adam()


def train_step(image_data, target):
    with tf.GradientTape() as t:
        prediction_result = yolo(image_data, training=True)
        giou_larray=0
        conf_larray=0
        prob_larray=0
        total_loss=0

        for itr in range(0,3):
            conv, pred = prediction_result[itr*2], prediction_result[itr*2+1]
            loss_items = compute_loss(pred, conv, *target[itr], itr, CLASSES=TRAIN_CLASSES)
            giou_larray += loss_items[0]
            conf_larray += loss_items[1]
            prob_larray += loss_items[2]

        total_loss+=giou_larray 
        total_loss+=conf_larray 
        total_loss+=prob_larray

        gradients = t.gradient(total_loss, yolo.trainable_variables)
        optimizer.apply_gradients(zip(gradients, yolo.trainable_variables))

        global_steps.assign_add(1)
        if global_steps < warmup_steps and not TRAIN_TRANSFER:
            lr = global_steps / warmup_steps * TRAIN_LR_INIT
        else:
            lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END)*((1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))

        optimizer.lr.assign(lr.numpy())

        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_larray, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_larray, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_larray, step=global_steps)
        writer.flush()
        
    return global_steps.numpy(), optimizer.lr.numpy(), giou_larray.numpy(), conf_larray.numpy(), prob_larray.numpy(), total_loss.numpy()

validate_writer = tf.summary.create_file_writer(logdir)#"./validate_log")

def validate_step(image_data, target):
    with tf.GradientTape() as t:
        prediction_result = yolo(image_data, training=False)
        
        giou_larray=0
        conf_larray=0
        prob_larray=0
        total_loss=0

        #Optimize
        for itr in range(0,3):
            conv, pred = prediction_result[itr*2], prediction_result[itr*2+1]
            loss_items = compute_loss(pred, conv, *target[itr], itr, CLASSES=TRAIN_CLASSES)
            prob_larray += loss_items[2]
            conf_larray += loss_items[1]
            giou_larray += loss_items[0]
            
        total_loss+=giou_larray 
        total_loss+=conf_larray 
        total_loss+=prob_larray

        with validate_writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("validate_loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("validate_loss/giou_loss", giou_larray, step=global_steps)
            tf.summary.scalar("validate_loss/conf_loss", conf_larray, step=global_steps)
            tf.summary.scalar("validate_loss/prob_loss", prob_larray, step=global_steps)
        validate_writer.flush()
        
    return giou_larray.numpy(), conf_larray.numpy(), prob_larray.numpy(), total_loss.numpy()


best_val_loss = 1000

for eph in range(TRAIN_EPOCHS):

    count = 0
    giou = 0 
    conf = 0
    prob = 0
    total = 0

    #Training
    for img_data1, targ in training_dataset:
        result1 = train_step(img_data1, targ)
        current_step = result1[0]%steps

    #Testing
    for img_data2, targt in testing_dataset:
        result2 = validate_step(img_data2, targt)
        count += 1
        total += result2[3]
        prob += result2[2]
        conf += result2[1]
        giou += result2[0]

    total_loss = total_val/count
    print("\n total_val_loss:{:7.2f}\n\n".format(total_loss))

    if save_best_only and best_val_loss>total_loss:
        yolo.save_weights("./checkpoints/yolov3_custom")
        best_val_loss = total_loss
    elif save_checkpoints and not save_checkpoints:
        yolo.save_weights("./checkpoints/yolov3_custom"+"_val_loss_{:7.2f}".format(total_loss))
    elif not save_best_only and not save_checkpoints:
        yolo.save_weights("./checkpoints/yolov3_custom")
