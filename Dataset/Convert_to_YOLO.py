import numpy as np
import cv2
import os

# Blue color in BGR 
color = (255, 0, 0)   
# Line thickness of 2 px 
thickness = 2
# Load an color image in grayscale
T = 'Train' 
root_path = os.getcwd() + '\\'+T+'\\Images\\'
num = '2'
wf = open(os.getcwd()+"\\"+T+"\\labels.txt", "a")
for numm in range(300):
    num = str(numm+1)
    img = cv2.imread('Images\\'+num+'.jpeg')
    f = open(os.getcwd()+"\\"+T+"\\Labels\\"+num+".txt", "r")
    n = int(f.readline())
    path = root_path + num + ".jpeg" 
    # print(path,end = " ")
    
    for i in range(n):
        pos = f.readline()[:-1].split(" ")
        x1 = pos[0]
        y1 = pos[1]
        x2 = pos[2]
        y2 = pos[3]
        # x = (x2+x1)/2
        # y = (y2+y1)/2
        # w = x2-x1
        # h = y2-y1
        # start = (x1,y1)
        # end = (x2,x2)
        # img = cv2.rectangle(img, start, end, color, thickness)
        path += " "+x1+","+y1+","+x2+","+y2+","+"0"
        # print(x1+","+y1+","+x2+","+y2+","+"0",end = " ")
    print(path)
    wf.write(path+"\n")
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
wf.close()