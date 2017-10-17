#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:54:18 2017

@author: ujjwalx
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
import cv2
import os

def histogramEqualize(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img

startTime = time.time() 
calibration_scales = []
fileList = sorted(os.listdir('calibration'))

for idx,file in enumerate(fileList):
    # This loop loads image to be tested.
    input_image = cv2.imread('calibration/' + file)
    img = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    #img = histogramEqualize(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hI,wI = img.shape[:2]
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    meth = methods[0]
    method = eval(meth)
    bestMatch = None
    for filename in os.listdir('templates_samesize'):
        # Loads Template to be matched

        input_template = cv2.imread('templates_samesize/'+ filename)
        template = cv2.cvtColor(input_template,cv2.COLOR_RGB2BGR)
        template = cv2.cvtColor(template,cv2.COLOR_BGR2HSV)
        #template = histogramEqualize(template)
        hT, wT = template.shape[:2]
        
        for scale in np.linspace(1,0.2,20):
            # Scale the original image to new size
            scaled_img = cv2.resize(img,(int(scale*wI),int(scale*hI)))
            r = img.shape[1] / float(scaled_img.shape[1])
            
            if scaled_img.shape[0] < hT or scaled_img.shape[1] < wT:
                break
            
            res = cv2.matchTemplate(scaled_img,template,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if bestMatch is None or max_val > bestMatch[0]:
                bestMatch = tuple([max_val,max_loc,filename,r])       
    
    (_,maxLoc,template_name,r) = bestMatch
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + wT) * r), int((maxLoc[1] + hT) * r))
        
    calibration_scales.append(tuple([(startX,startY),(endX,endY),r,file,template_name]))    

print ("--This run took %0.2f seconds--" %(time.time()-startTime))