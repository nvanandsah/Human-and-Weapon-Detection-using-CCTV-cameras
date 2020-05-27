import cv2
import time
import random
import colorsys
import numpy as np
import tensorflow as tf
from yolov3.configs import *
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime
from google.cloud import storage
from firebase import firebase
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/navne/OneDrive/Documents/Codes/EDGEAI/TF_Yolo_Gun_Detection/fir-5f27b-firebase-adminsdk-470w3-16931e232d.json"
firebase = firebase.FirebaseApplication('https://fir-5f27b.firebaseio.com/')
client = storage.Client()
bucket = client.get_bucket('fir-5f27b.appspot.com')
# posting to firebase storage
imageBlob = bucket.blob("/")
# Fetch the service account key JSON file contents
cred = credentials.Certificate('C:/Users/navne/OneDrive/Documents/Codes/EDGEAI/TF_Yolo_Gun_Detection/fir-5f27b-firebase-adminsdk-470w3-16931e232d.json')
# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://fir-5f27b.firebaseio.com/'
})
flag_box = False
def load_yolo_weights(model, weights_file):
    with open(weights_file, 'rb') as f:
        #major, minor, revision, seen, _ = np.fromfile(f, dtype=np.int32, count=5)

        c1 = 0; c2 = 0
        while c2 < 75:
            if c1 == 0:
                bn_layer_n = 'batch_norm'
            elif c1>0:
                bn_layer_n = 'batch_norm_%d' %j

            if c2 == 0:
                conv_layer_n = 'conv2d'
            elif c2>0:
                conv_layer_n = 'conv2d_%d' %i
                
            conv_layer = model.get_layer(conv_layer_n)
            filters = conv_layer.filters
            ker_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]

            if c2 in [58, 66, 74]:
                conv_bias = np.fromfile(f, dtype=np.float32, count=filters)
            else:
                bn_weights = np.fromfile(f, dtype=np.float32, count=4 * filters)
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = model.get_layer(bn_layer_n)
                c1+= 1

            conv_shape = (filters, in_dim, ker_size, ker_size)
            conv_weights = np.fromfile(f, dtype=np.float32, count=np.product(conv_shape))
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if c2 in [58, 66, 74]:
                conv_layer.set_weights([conv_weights, conv_bias])
            else:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)

            c2+=1

        assert len(f.read()) == 0, 'failed to read all data'


def read_class_names(clsFile):
    class_names = dict()
    with open(clsFile, 'r') as f1:
        for id, name in enumerate(f1):
            class_names[id] = name.strip('\n')
    return class_names

def image_preprocess(image, target_size, gt_boxes=None):
    iht, iwt    = target_size
    ht,  wt, _  = image.shape

    s1=iwt/wt
    s2=iht/ht
    s=min(s1,s2)

    nw = int(s * wt)
    nh = int(s * ht)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[iht, iwt, 3], fill_value=128.0)
    dw = (iwt - nw) // 2
    dh = (iht - nh) // 2

    dh2=nh+dh
    dw2=nw+dw
    image_paded[dh:dh2, dw:dw2, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is not None:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * s + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * s + dh
        return image_paded, gt_boxes
    else:
        return image_paded


def draw_bbox(image, bboxes, CLASSES=YOLO_COCO_CLASSES, show_label=True, show_confidence = True, Text_colors=(255,255,0), rectangle_colors=''):   
    i_h, i_w, _ = image.shape
    NUM_CLASS = read_class_names(CLASSES)
    n_class = len(NUM_CLASS)
    
    hsv_tuples = [(1.0 * x / n_class, 1., 1.) for x in range(n_class)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    flag_box = False
    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != ''else colors[class_ind]
        bbox_thick = int(0.6 * (i_h + i_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        if show_label:
            
            score_str = f' {score:.2f}' if show_confidence else '' 
            label = f'{NUM_CLASS[class_ind]}' + score_str
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale, thickness=bbox_thick)
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)
    print(str(len(bboxes)) +" "+ str(len(bboxes)!=0))
    if(len(bboxes)!=0):
        flag_box = True    
    print(flag_box)
    return flag_box,image


def bboxes_iou(box1, box2):
    box1 = np.array(box1)
    box2 = np.array(box2)

    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    box_area  = box1_area + box2_area

    l_up       = np.maximum(box2[..., :2], box1[..., :2])
    r_down    = np.minimum(box2[..., 2:],box1[..., 2:])
    diff = r_down - l_up

    intersection = np.maximum(0.0, diff)
    intersection_area    = intersection[..., 0] * intersection[..., 1]
    
    union_area    = box_area - intersection_area

    div_area = intersection_area / union_area
    ious     = np.maximum(1.0 * div_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    img_class = list(set(bboxes[:, 5]))
    algo_bboxes = list()

    for itr in range(len(img_class)):
        cl = img_class[itr]
        cl_bboxes = bboxes[(bboxes[:, 5] == cl)]
        while len(cl_bboxes) > 0:
            index_max = np.argmax(cl_bboxes[:, 4])
            bbox_nms = cl_bboxes[index_max]
            algo_bboxes.append(bbox_nms)
            cl_bboxes = np.concatenate([cl_bboxes[: index_max], cl_bboxes[index_max + 1:]])
            iou = bboxes_iou(bbox_nms[np.newaxis, :4], cl_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert algo in ['nms', 'soft-nms']

            if algo != 'nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            if algo != 'soft-nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            cl_bboxes[:, 4] = cl_bboxes[:, 4] * weight
            score_mask = cl_bboxes[:, 4] > 0.
            cl_bboxes = cls_bboxes[score_mask]

    return algo_bboxes


def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_1 = pred_bbox[:, 0:4]
    pred_2 = pred_bbox[:, 4]
    pred_3 = pred_bbox[:, 5:]

    pred_diff = pred_1[:, :2] - pred_1[:, 2:]
    pred_sum  = pred_1[:, :2] + pred_1[:, 2:]
    pred_4 = np.concatenate([pred_diff * 0.5, pred_sum* 0.5], axis=-1)

    original_h, original_w = original_image.shape[:2]
    div1_org = input_size / original_w
    div2_org = input_size / original_h
    resize_ratio = min(div1_org,div2_org )

    diff_ratio=input_size - resize_ratio
    dw = ( diff_ratio * original_w) / 2
    dh = ( diff_ratio * original_h) / 2

    pred_4[:, 0::2] = 1.0 * (pred_4[:, 0::2] - dw) / resize_ratio
    pred_4[:, 1::2] = 1.0 * (pred_4[:, 1::2] - dh) / resize_ratio

    max_pred = np.maximum(pred_4[:, :2], [0, 0])
    min_pred = np.minimum(pred_4[:, 2:], [original_w - 1, original_h - 1])
    pred_4 = np.concatenate([max_pred,min_pred], axis=-1)

    false_mask = np.logical_or((pred_4[:, 0] > pred_4[:, 2]), (pred_4[:, 1] > pred_4[:, 3]))
    pred_4[false_mask] = 0

    #Discard box
    bbox_s = np.sqrt(np.multiply.reduce(pred_4[:, 2:4] - pred_4[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bbox_s), (bbox_s < valid_scale[1]))

    #low scored
    classes = np.argmax(pred_3, axis=-1)
    scores = pred_2 * pred_3[np.arange(len(pred_4)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_4[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


def detect_image(YoloV3, image_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
    o_image      = cv2.imread(image_path)
    o_image      = cv2.cvtColor(o_image, cv2.COLOR_BGR2RGB)
    
    img = image_preprocess(np.copy(o_image), [input_size, input_size])
    img = tf.expand_dims(img, 0)

    pred_bbox = YoloV3.predict(img)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    
    bboxes = postprocess_boxes(pred_bbox, o_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')

    flag_box,image = draw_bbox(o_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)

    cv2.imwrite(output_path, image)
    print(flag_box)

    if(flag_box):
        # imagePath = "C:/Users/navne/OneDrive/Documents/Codes/EDGEAI/TF_Yolo_Gun_Detection/Dataset/Testing_out/1.jpg"
        millis = int(round(time.time() * 1000))
        imageBlob = bucket.blob(str(millis)+".jpg")
        imageBlob.upload_from_filename(output_path)
        ref = db.reference('/')
        ref.push({
            'TimeStamp': str(datetime.now()),
            'Location': "Store1",
            'ImageURL':"gs://fir-5f27b.appspot.com/"+str(millis)+".jpg"
        })
    if show:
        cv2.imshow("predicted image", image)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return
