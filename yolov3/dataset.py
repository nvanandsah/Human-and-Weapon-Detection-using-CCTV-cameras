from yolov3.utils import read_class_names, image_preprocess
from yolov3.yolov3 import bbox_iou
from yolov3.configs import *

import os
import cv2
import random
import numpy as np
import tensorflow as tf


class Dataset(object):
    # implement Dataset here
    def __init__(self, dataset_type):

        self.num_classes = len(self.classes)
        self.train_input_sizes = TRAIN_INPUT_SIZE
        self.classes = read_class_names(TRAIN_CLASSES)
        self.strides = np.array(YOLO_STRIDES)
        

        temp = (np.array(YOLO_ANCHORS).T/self.strides).T
        self.anchors = temp

        self.anchor_per_scale = YOLO_ANCHOR_PER_SCALE
        self.max_bbox_per_scale = YOLO_MAX_BBOX_PER_SCALE


        if (dataset_type == 'train'):
            self.data_aug = TRAIN_DATA_AUG  
        else :
            self.data_aug  = TEST_DATA_AUG
        

        if (dataset_type == 'train'):
            self.input_sizes = TRAIN_INPUT_SIZE  
        else :
            self.input_sizes  = TEST_INPUT_SIZE


        if (dataset_type == 'train'):
            self.annot_path  = TRAIN_ANNOT_PATH  
        else :
            self.annot_path  = TEST_ANNOT_PATH


        if (dataset_type == 'train'):
            self.batch_size  = TRAIN_BATCH_SIZE  
        else :
            self.batch_size  = TEST_BATCH_SIZE


        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)

        temp = self.num_samples / self.batch_size
        self.num_batchs = int(np.ceil(temp))

        self.batch_count = 0


    def load_annotations(self, dataset_type):

        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            v = len(line.strip().split()[1:])
            check = v != 0
            ann = [line.strip() for line in txt if check==True]

        np.random.shuffle(ann)
        return ann


    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            c = 4

            self.train_input_size = random.choice([self.train_input_sizes])

            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3), dtype=np.float32)

            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, (c+1) + self.num_classes), dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, (c+1) + self.num_classes), dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, (c+1) + self.num_classes), dtype=np.float32)

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, c), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, c), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, c), dtype=np.float32)

            num = 0
            if self.batch_count < self.num_batchs:

                while num < self.batch_size:

                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                    mid = self.batch_count * self.batch_size
                    curr = mid + num
                    if (curr >= self.num_samples): 
                        curr -= self.num_samples

                    annotation = self.annotations[curr]

                    image, bboxes = self.parse_annotation(annotation)
                    
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes

                    batch_image[num, :, :, :] = image

                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    
                    num += 1

                self.batch_count += 1
                small = batch_label_sbbox, batch_sbboxes
                medium  = batch_label_mbbox, batch_mbboxes
                large  = batch_label_lbbox, batch_lbboxes

                ans = (small, medium, large)

                return batch_image, ans
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_crop(self, image, bboxes):
        bar = 0.5
        if (random.random() < bar):
            h, w, _ = image.shape

            X = np.min(bboxes[:, 0:2], axis=0)
            Y = np.max(bboxes[:, 2:4], axis=0)
            
            max_bbox = np.concatenate([X, Y], axis=-1)

            max_l = max_bbox[0]
            max_u = max_bbox[1]
            max_r = w - max_bbox[2]
            max_d = h - max_bbox[3]


            temp1 = int(max_bbox[0] - random.uniform(0, max_l))
            temp2 = int(max_bbox[1] - random.uniform(0, max_u))
            temp3 =int(max_bbox[2] + random.uniform(0, max_r))
            temp4 = int(max_bbox[3] + random.uniform(0, max_d))

            xmin = max(0, temp1)
            ymin = max(0, temp2)
            xmax = max(w, temp3)
            ymax = max(h, temp4)

            image = image[ymin : ymax, xmin : xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - ymin

        return image, bboxes

   


    def random_horizontal_flip(self, image, bboxes):
        bar = 0.5
        if (random.random() < bar):
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, bboxes


    def random_translate(self, image, bboxes):
        bar =0.5
        if random.random() < bar:
            h, w, _ = image.shape

            X = np.min(bboxes[:, 0:2], axis=0)
            Y = np.max(bboxes[:, 2:4], axis=0)

            maximum_bbox = np.concatenate([X, Y], axis=-1)

            max_l = maximum_bbox[0]
            max_u = maximum_bbox[1]
            max_r = w - maximum_bbox[2]
            max_d = h - maximum_bbox[3]

            start = -(max_l - 1)
            end = (max_r - 1)

            tx = random.uniform(start, end)

            start_1 = -(max_u - 1)
            end_1 = (max_d - 1)

            ty = random.uniform(start_1, end_1)

            M = np.array([[1, 0, tx], [0, 1, ty]])

            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes


    def parse_annotation(self, annotation):
        line = annotation.split()
        image_path = line[0]

        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)

        image = cv2.imread(image_path)
        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image, bboxes = image_preprocess(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
        return image, bboxes



    def preprocess_true_boxes(self, bboxes):

        t = 5 + self.num_classes
        a = self.train_output_sizes[i]
        b = self.train_output_sizes[i]

        one = (a, b, self.anchor_per_scale, t)

        l = [np.zeros(one) for i in range(3)]

        b_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]

        bbox_count = np.zeros((3,))

        for bbox in bboxes:

            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            a = 1.0 / self.num_classes
            ud = np.full(self.num_classes, a)
            deta = 0.01

           
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            iou = []

            check = False

            s_onehot = onehot * (1 - deta) + deta * ud



            for i in range(3):

                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = bbox_iou(bbox_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_scaled[i, 0:2]).astype(np.int32)

                    l[i][yind, xind, iou_mask, 4:5] = 1.0
                    l[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    l[i][yind, xind, iou_mask, 5:] = s_onehot
                    l[i][yind, xind, iou_mask, :] = 0
                    

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    b_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    check = True

            if not check:

                anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)

                x = anchor_ind / self.anchor_per_scale
                detect = int(x)

                y = anchor_ind % self.anchor_per_scale
                anchor = int(y)

                z = bbox_scaled[detect, 0:2]

                xind, yind = np.floor(z).astype(np.int32)

                l[detect][yind, xind, anchor, 5:] = s_onehot
                l[detect][yind, xind, anchor, 0:4] = bbox_xywh
                l[detect][yind, xind, anchor, 4:5] = 1.0
                l[detect][yind, xind, anchor, :] = 0

                a = bbox_count[detect]

                temp_num = (a % self.max_bbox_per_scale)

                bbox_ind = int(temp_num)

                b_xywh[detect][bbox_ind, :4] = bbox_xywh

                bbox_count[detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label

        sbboxes, mbboxes, lbboxes = b_xywh

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs
