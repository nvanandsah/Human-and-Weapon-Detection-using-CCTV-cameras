import numpy as np
import cv2 
import copy

#img2 = cv2.imread('navneet.jpeg')
import numpy as np
import cv2
from keypoint import keypoint
from detection import detection

img = cv2.imread('na1.jpeg')
img2 = cv2.imread('nv1.jpeg')


scale = 0.3
img = cv2.resize(img,(int(scale*img.shape[1]),int(scale*img.shape[0])))
img2 = cv2.resize(img2,(int(scale*img2.shape[1]),int(scale*img2.shape[0])))
original_image = np.concatenate((img, img2), axis=1)

img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

imgC = cv2.split(img)
img2C = cv2.split(img2)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(imgC[0])
cl2 = clahe.apply(img2C[0])

imgC[0] = cl1
img2C[0] = cl2

img1 = cv2.merge(imgC)
img2 = cv2.merge(img2C)

img1 = cv2.cvtColor(img1, cv2.COLOR_LAB2BGR)
img2 = cv2.cvtColor(img2, cv2.COLOR_LAB2BGR)

clahe_image = np.concatenate((img1, img2), axis=1)
cv2.imshow('Original',original_image)
cv2.imshow('After Clahe',clahe_image)
cv2.imwrite('original.jpg', original_image)
cv2.imwrite('clahe.jpg', clahe_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



box_body1, body_face1, img1_face = detection(img1)
box_body2, body_face2, img2_face = detection(img2)
face_detection = np.concatenate((img1_face, img2_face), axis=1)
cv2.imshow('Face Detection',face_detection)
cv2.imwrite('face_detection.jpg', face_detection)
cv2.waitKey(0)
cv2.destroyAllWindows()

keypoint(img1,img2,box_body1,box_body2, 0)

def blendImg(img1,img2,c1,c2):
    difference = float(c2)-float(c1)
    step = 1.0 / difference    
    row, col, _ = img1.shape
    img3 = copy.deepcopy(img1)
    for r in range(row):
        i = 1
        for c in range(col):
            if (c > c2):
                img3[r][c] = img2[r][c]
            elif (c >= c1 and c<= c2):
                i += 1
                img3[r][c] = (1 - step * i) * img1[r][c] + (step * i) * img2[r][c]
    return img3
    
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

