import numpy as np
import cv2 
import copy

#img = cv2.imread('navdha.jpeg')
#img2 = cv2.imread('navneet.jpeg')

def detection(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3)
    
    (x,y,w,h) = faces[0]
    # img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    img = copy.deepcopy(img)
    img_body = cv2.rectangle(img,(x-w,y-h),(x+2*w,img.shape[0]),(255,0,0),2)
    
    # cv2.imshow('img',img)
    # cv2.waitKey(0)    
    (x,y,w,h) = faces[0]
    return ((x-w,y,x+2*w,img.shape[0]),(x,y,x+w,y+h), img_body)
