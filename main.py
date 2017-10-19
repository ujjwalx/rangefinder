#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import cv2

LOWER_PINK = np.array([157, 100, 100])
UPPER_PINK = np.array([177, 255, 255])

LOWER_GREEN = np.array([65, 60, 60])
UPPER_GREEN = np.array([90, 150, 150])

LOWER_BLUE = np.array([91, 100, 100])
UPPER_BLUE = np.array([111, 255, 255])

LOWER_YELLOW = np.array([17, 100, 100])
UPPER_YELLOW = np.array([37, 255, 255])


def histogram_equalize(img):

    # Performs Histogram Equalization by converting to YCrCb color space.
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


def remove_non_informative_regions(img):

    # Remove empirically observed non-informative regions
    height, width = img.shape[:2]
    croppedImage = img[80:(height - 90), 0:width]
    return croppedImage


startTime = time.time()
detections = []
fileList = sorted(os.listdir('calibration'))

for idx, file in enumerate(fileList):

    # This loop loads image to be tested.
    img = cv2.imread('calibration/' + file)
    img = remove_non_informative_regions(img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #img = histogramEqualize(img)
    hI, wI = img.shape[:2]

    # Threshold the HSV image to get only pink color.
    mask = cv2.inRange(img_hsv, LOWER_PINK, UPPER_PINK)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    ret, gray = cv2.threshold(gray, 10, 255, 0)
    gray2 = gray.copy()
    mask = np.zeros(gray.shape, np.uint8)
    _, contours, hier = cv2.findContours(
        gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pinkLocations = []
    for cnt in contours:
        if 40 < cv2.contourArea(cnt) < 5000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.3 * peri, True)
            # cv2.drawContours(img,[cnt],0,(0,255,0),2)
            # cv2.drawContours(mask,[cnt],0,255,-1)
            x, y, w, h = cv2.boundingRect(cnt)
            pinkLocations.append(tuple([x, y, w, h]))

            # Enable for Debugging
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    for loc in pinkLocations:

        (x, y, w, h) = loc
        patches = [tuple([x, y - h, w, h, -1]), tuple([x, y + h, w, h, 1])]

        for patch in patches:
            (x, y, w, h, orientation) = patch
            out = img_hsv[y:y + h, x:x + w]
            avg_hsv = np.average(np.average(out, axis=0), axis=0)
            if (LOWER_GREEN[0] <= avg_hsv[0] <= UPPER_GREEN[0] and
                LOWER_GREEN[1] <= avg_hsv[1] <= UPPER_GREEN[1] and
                    LOWER_GREEN[2] <= avg_hsv[2] <= UPPER_GREEN[2]):

                if (orientation == 1):
                    detections.append(tuple([file, "Pink_Green"]))
                else:
                    detections.append(tuple([file, "Green_Pink"]))

            if (LOWER_BLUE[0] <= avg_hsv[0] <= UPPER_BLUE[0] and
                LOWER_BLUE[1] <= avg_hsv[1] <= UPPER_BLUE[1] and
                    LOWER_BLUE[2] <= avg_hsv[2] <= UPPER_BLUE[2]):

                if (orientation == 1):
                    detections.append(tuple([file, "Pink_Blue"]))
                else:
                    detections.append(tuple([file, "Blue_Pink"]))

            if (LOWER_YELLOW[0] <= avg_hsv[0] <= UPPER_YELLOW[0] and
                LOWER_YELLOW[1] <= avg_hsv[1] <= UPPER_YELLOW[1] and
                    LOWER_YELLOW[2] <= avg_hsv[2] <= UPPER_YELLOW[2]):

                if (orientation == 1):
                    detections.append(tuple([file, "Pink_Yellow"]))
                else:
                    detections.append(tuple([file, "Yellow_Pink"]))


print("--This run took %0.2f seconds--" % (time.time() - startTime))
