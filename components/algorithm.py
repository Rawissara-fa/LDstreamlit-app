import streamlit as st
import matplotlib.pyplot as plt
import cv2
import os
import math
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from PIL import Image







# ------------------------------------
# Read images: my algorithm
# ------------------------------------
@st.cache_data
def load_data(data_dir):
    data_original = []
    
    for img_list in range (int(len(data_dir))):
        
        # st.write(int(len(data_dir)))
        for img in range (int(len(data_dir))):
            # st.write(img)
            img_arr = cv2.imread(data_dir[img])
            
            # convert BGR to RGB format for data_original
            # img_arr = image.resize((1600, 1200))
            resized_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
            # st.write(resized_arr)
            
            mix_imgs = []
            aug_img = np.array(
                [resized_arr for _ in range(50)], dtype=np.uint8)
            for i in range(10):
                mix_imgs.append(aug_img[i])
            for ii in range(9):
                mix_img = mix_imgs[0]+mix_imgs[ii+1]
            data_original.append([img_arr,mix_img, img])
            # st.write(data_original[0][2])

        return data_original


## ------------------------------------------------------------------------------
## Check Surface of LD chip

def CheckSurfaceLDArea(image_LD, Surface_limit, contrast_limit):
    
    # image_LD.shape_paper = (V:100, H:350, D:3)
    image_gray = cv2.cvtColor(image_LD, cv2.COLOR_BGR2GRAY)
    _, image_th = cv2.threshold(image_gray, contrast_limit, 1, cv2.THRESH_BINARY)
    pixel_V = int(1/5*image_th.shape[0]) 
    pixel_H = image_th.shape[1]
    criteria = int(pixel_V)

    valuebox = np.zeros((1,1))
    sumx_top = np.sum(image_th[0:pixel_V, 0:pixel_H], axis=0)

    counttopx = 0
    for nx in range (0, pixel_H):
        if (sumx_top[nx] < criteria):
            counttopx += 1
    valuebox[0,0] = counttopx
    
    if (valuebox[0][0] <= Surface_limit):
        result = "OK"
    else:
        result = "NG"
    
    return result

## ------------------------------------------------------------------------------
## Check LD area of LD chip

def CheckLDArea(image_LD, LDarea_limit, contrast_limit):
        
    # image_LD.shape_paper = (V:100, H:350, D:3)
    image_gray = cv2.cvtColor(image_LD, cv2.COLOR_BGR2GRAY)
    _, image_th = cv2.threshold(image_gray, contrast_limit, 1, cv2.THRESH_BINARY)

    
    valueboxH = np.zeros((1,2))
    valueboxV = np.zeros((1,2))
    count_HL = 0 ;count_HR = 0
    count_VL = 0 ;count_VR = 0
    
    pixel_V = image_th.shape[0]
    pixel_H = image_th.shape[1]
    criteria_V = int(3/7*pixel_H)
    criteria_H = pixel_V

    ##Horizontal axis   
    ##Box X-Left
    sumx_HL = np.sum(image_th[0:pixel_V, 0:int(3/7*pixel_H)], axis=0)
    ##Box X-Right
    sumx_HR = np.sum(image_th[0:pixel_V, int(4/7*pixel_H):pixel_H], axis=0)
    
    for nX in range (5, int(3/7*pixel_H)):
        if (sumx_HL[nX] < criteria_H-5):
            count_HL += 1
        if (sumx_HR[nX] < criteria_H-1):
            count_HR += 1
    valueboxH[0,0] = count_HL
    valueboxH[0,1] = count_HR
        
    ##vertical axis   
    ##Box Y-Left
    sumy_VL = np.sum(image_th[0:pixel_V, 0:int(3/7*pixel_H)], axis=1)
    ##Box Y-Right
    sumy_VR = np.sum(image_th[0:pixel_V, int(4/7*pixel_H):pixel_H], axis=1)
    
    for nY in range (0, pixel_V):
        if (sumy_VL[nY] < criteria_V-5):
            count_VL += 1
        if (sumy_VR[nY] < criteria_V):
            count_VR += 1
    valueboxV[0,0] = count_VL
    valueboxV[0,1] = count_VR
    
    # print(valueboxH[0])
    # print(valueboxV[0])
    # print("sumx_HL ------------------------", sumx_HL[0:25])
    
    if ((valueboxH[0][0] <= LDarea_limit) & (valueboxH[0][1] <= LDarea_limit) 
        & (valueboxV[0][0] <= LDarea_limit) & (valueboxV[0][1] <= LDarea_limit)):
        result = "OK"
    else:
        result = "NG"   
    
    return result

## ------------------------------------------------------------------------------
## Check Conner area of LD chip

def CheckConnerArea(image_LD, ConnerArea_limit, contrast_limit):
        
    # image_LD.shape_paper = (V:100, H:350, D:3)
    image_gray = cv2.cvtColor(image_LD, cv2.COLOR_BGR2GRAY)
    _, image_th = cv2.threshold(image_gray, contrast_limit, 1, cv2.THRESH_BINARY)
    valueboxH = np.zeros((1,2))
    valueboxV = np.zeros((1,2))
    count_HL = 0 ;count_HR = 0 
    count_VL = 0 ;count_VR = 0 
    
    pixel_V = image_th.shape[0]
    pixel_H = image_th.shape[1]
    criteria_H = int(1/5*pixel_V)
    criteria_V = int(1/7*pixel_H)

    ##Horizontal axis   
   
    ##Box X-Left
    sumx_HL = np.sum(image_th[0:int(1/5*pixel_V), 0:pixel_H], axis=0)
    ##Box X-Right
    sumx_HR = np.sum(image_th[0:int(1/5*pixel_V), 0:pixel_H], axis=0)
    
    for nX in range (0, pixel_H-1):
        if (sumx_HL[nX] < criteria_H):
            count_HL += 1
        if (sumx_HR[nX] < criteria_H):
            count_HR += 1
    valueboxH[0,0] = count_HL
    valueboxH[0,1] = count_HR
        
    ##vertical axis   
    ##Box Y-Left
    sumy_VL = np.sum(image_th[0:pixel_V, 0:int(1/7*pixel_H)], axis=1)
    ##Box Y-Right
    sumy_VR = np.sum(image_th[0:pixel_V, int(6/7*pixel_H):pixel_H], axis=1)
    
    for nY in range (0,pixel_V):
        if (sumy_VL[nY] < criteria_V):
            count_VL += 1
        if (sumy_VR[nY] < criteria_V):
            count_VR += 1       
    valueboxV[0,0] = count_VL
    valueboxV[0,1] = count_VR
    
    if ((valueboxH[0][0] <= ConnerArea_limit) & (valueboxH[0][1] <= ConnerArea_limit) 
        & (valueboxV[0][1] <= ConnerArea_limit) & (valueboxV[0][1] <= ConnerArea_limit)):
        result = "OK"
    else:
        result = "NG"  
    
    return result

## ------------------------------------------------------------------------------
## Check AR area of LD chip

def CheckARArea(image_LD, ARArea_limit, contrast_limit):
    
    # image_LD.shape_paper = (V:100, H:350, D:3)
    image_gray = cv2.cvtColor(image_LD, cv2.COLOR_BGR2GRAY)
    _, image_th = cv2.threshold(image_gray, contrast_limit, 1, cv2.THRESH_BINARY)
    pixel_V = image_th.shape[0]
    pixel_H = image_th.shape[1]
    criteria_H = int(1/10*pixel_V)-1
    criteria_V = int(1/7*pixel_H)

    valuebox = np.zeros((1,2))
    # sumx_AR = np.sum(image_th[int(4/5*pixel_V):pixel_V, int(3/7*pixel_H):int(5/7*pixel_H)], axis=0)
    # sumy_AR = np.sum(image_th[int(4/5*pixel_V):pixel_V, int(3/7*pixel_H):int(5/7*pixel_H)], axis=1)
    sumx_AR = np.sum(image_th[int(9/10*pixel_V):pixel_V-1, int(3/7*pixel_H):int(4/7*pixel_H)], axis=0)
    sumy_AR = np.sum(image_th[int(9/10*pixel_V):pixel_V-1, int(3/7*pixel_H):int(4/7*pixel_H)], axis=1)
    
    countx = 0
    county = 0
    for nx in range (0, criteria_V):
        if (sumx_AR[nx] < criteria_H):
            countx += 1
            
    for ny in range (0, criteria_H):
        if (sumy_AR[ny] < criteria_V):
            county += 1
            
    valuebox[0,0] = countx
    valuebox[0,1] = county
    
    if ((valuebox[0][0] <= ARArea_limit) & (valuebox[0][1] <= ARArea_limit)):             
        result = "OK"
    else:
        result = "NG"
    
    # print("AR =", criteria_H)
    # print(countx, county, result)
    
    return result