import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy

def keypoint(img1, img2,bb1,bb2, ddd):
    c1 = bb1[1]
    c2 = bb1[3]
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create()
    
    if(ddd == 0):
        kp_img1, des_img1 = sift.detectAndCompute(img1,None)
        kp_img2, des_img2 = sift.detectAndCompute(img2,None)
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    if(ddd == 1):
        kp_img1, des_img1 = surf.detectAndCompute(img1,None)
        kp_img2, des_img2 = surf.detectAndCompute(img2,None)
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    if(ddd == 2):
        kp_img1, des_img1 = orb.detectAndCompute(img1,None)
        kp_img2, des_img2 = orb.detectAndCompute(img2,None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    matches = bf.match(des_img1, des_img2)
    new_matches = []
    matches = cleanmatches(matches, kp_img1, kp_img2, bb1, bb2, new_matches)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    #print(matches[:5])
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp_img1,img2,kp_img2,matches[:20], None, flags=2)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    plt.imshow(img3),plt.show()

    point_img1 = []
    point_img2 = []
    print(len(matches))
    i=1
    for match in matches:
        i=i+1
        point_img1.append(kp_img1[match.queryIdx].pt)
        point_img2.append(kp_img2[match.trainIdx].pt)
        if(i>25):
            break
    point_img1 = np.array(point_img1)
    point_img2 = np.array(point_img2)
    H = cv2.findHomography(point_img2, point_img1, cv2.RANSAC)
    print(type(H[0]))
    warped_image = cv2.warpPerspective(img2, H[0], (img1.shape[1], img1.shape[0]))
    print(img1.shape[:2][1], img1.shape[:2][0], img1.shape)
    warp_image = np.concatenate((img1, warped_image), axis=1)
    cv2.imwrite('warped_image.jpg',warp_image)
    cv2.imshow('warped back image', warp_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(c1,c2)
    img3 = blendImg(img1,warped_image,c1,c2)
    cv2.imwrite('Blended.jpg',img3)
    cv2.imshow('Blended image', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #def alpha_blend(img_left, img_right, col_start, col_end):

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
    
def cleanmatches(matches, kp1, kp2, bb1, bb2, newmatch):
    for match in matches:
        kp = kp1[match.queryIdx]
        if (check(kp, bb1) == False): 
            continue
        if (check(kp, bb2) == False): 
            continue   
        kp = kp2[match.trainIdx]
        if (check(kp, bb1) == False): 
            continue
        if (check(kp, bb2) == False): 
            continue  
        newmatch.append(match)
    return newmatch

def check(kp, bb):
    x,y = kp.pt[0],kp.pt[1]
    xtl, ytl, xbr, ybr = bb
    flag = 0
    if ((x>xtl) and (x<xbr) and (y>ytl) and (y<ybr)):
            flag = 1
    if (flag == 1):
        return False
    return True


