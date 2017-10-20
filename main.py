#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time

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


def show_data(range_measurement, bearing_measurement):
    # Displays Range and Measurment Values

    print("Current Range from Landmark Detected is %0.2f meters" %
          range_measurement)
    print("Current Global Angle is %0.2f degrees" % bearing_measurement)

    return (0)

# Range and Estimation Calculations by Ward Heij.
# estimate distance from a landmark given the size of a detection


def estimate_range(landmark_location):

    # determine width and hieght of the patch
    #width = bottom_right[0] - top_left[0]
    #height = bottom_right[1] - top_left[1]

    # Unpack landmark location values
    (_, _, width, height, _) = landmark_location

    # set nominal values at 1 meter distance TODO: Check if this works correctly
    w_nominal = 47.16
    h_nominal = 107.32

    # estimate distance based on width/height ratios independently and average
    d1 = w_nominal / width
    d2 = h_nominal / height
    range_estimate = (d1 + d2) / 2.0

    return range_estimate


def get_landmark_angle_radians(signature):
    if signature == 1:
        angle = 5.4979
    if signature == 2:
        angle = 0
    if signature == 3:
        angle = 0.785398
    if signature == 4:
        angle = 2.35619
    if signature == 5:
        angle = 3.14159
    if signature == 6:
        angle = 3.92699

    return angle


def get_landmark_angle_degrees(signature):
    if signature == 1:
        angle = 315
    if signature == 2:
        angle = 0
    if signature == 3:
        angle = 45
    if signature == 4:
        angle = 135
    if signature == 5:
        angle = 180
    if signature == 6:
        angle = 225

    return angle

# estimate the bearing of the robot with respect to the landmark


def estimate_bearing(img, landmark_patch, range_measurement):

    fov = 60.9  # degrees, from Nao documentation

    hI, wI, channels = img.shape
    xcenterI = int(wI / 2)
    ycenterI = int(hI / 2)

    # Unpack landmark location values
    (xL, yL, wL, hL, signature) = landmark_location

    # find center of the observation, find distance from image center
    xcenterL = (xL + (xL + wL)) / 2.0
    ycenterL = (yL + (yL + hL)) / 2.0

    diff_angle = np.arctan(
        (abs(xcenterI - xcenterL) * 2 * np.tan(fov / 2)) / wI)

    landmark_angle = get_landmark_angle_degrees(signature)

    if (xcenterI - xcenterL >= 0):
        global_heading_angle = landmark_angle - diff_angle

    if (xcenterI - xcenterL < 0):
        global_heading_angle = landmark_angle + diff_angle

    #bearing_measurement = np.rad2deg(global_heading_angle)
    bearing_measurement = np.deg2rad(np.mod(global_heading_angle, 360))

    return bearing_measurement


def write_file(filename, detections):
    file = open(filename, 'w')
    for measurement in detections:
        (filename, _, detected_signature, range_measurement,
         bearing_measurement) = measurement
        file.write(str(filename) + '||' + str(detected_signature) + '||' + str(range_measurement) + '||'
                   + str(bearing_measurement) + '\n')


startTime = time.time()
detections = []
fileList = sorted(os.listdir('calibration'))

for file in fileList:

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

        # Orientation parameter '-1' suugests that the detected pink patch is on
        # the bottom of the landmark. Conversely '+1' suggests pink patch is at
        # the top.

        patches = [tuple([x, y - h, w, h, -1]), tuple([x, y + h, w, h, 1])]

        for patch in patches:

            landmark_location = None
            (x, y, w, h, orientation) = patch
            out = img_hsv[y:y + h, x:x + w]
            avg_hsv = np.average(np.average(out, axis=0), axis=0)

            if (LOWER_GREEN[0] <= avg_hsv[0] <= UPPER_GREEN[0] and
                LOWER_GREEN[1] <= avg_hsv[1] <= UPPER_GREEN[1] and
                    LOWER_GREEN[2] <= avg_hsv[2] <= UPPER_GREEN[2]):

                if (orientation == 1):
                    detected_signature = 5
                    landmark_location = tuple(
                        [x, y - h, w, 2 * h, detected_signature])
                    range_measurement = estimate_range(landmark_location)
                    bearing_measurement = estimate_bearing(img, landmark_location,
                                                           range_measurement)
                    detections.append(
                        tuple([file, "Pink_Green", detected_signature, range_measurement, bearing_measurement]))

                else:
                    detected_signature = 2
                    landmark_location = tuple(
                        [x, y, w, 2 * h, detected_signature])
                    range_measurement = estimate_range(landmark_location)
                    bearing_measurement = estimate_bearing(img, landmark_location,
                                                           range_measurement)
                    detections.append(
                        tuple([file, "Green_Pink", detected_signature, range_measurement, bearing_measurement]))

                show_data(range_measurement, bearing_measurement)

            if (LOWER_BLUE[0] <= avg_hsv[0] <= UPPER_BLUE[0] and
                LOWER_BLUE[1] <= avg_hsv[1] <= UPPER_BLUE[1] and
                    LOWER_BLUE[2] <= avg_hsv[2] <= UPPER_BLUE[2]):

                if (orientation == 1):
                    detected_signature = 6
                    landmark_location = tuple(
                        [x, y - h, w, 2 * h, detected_signature])
                    range_measurement = estimate_range(landmark_location)
                    bearing_measurement = estimate_bearing(img, landmark_location,
                                                           range_measurement)
                    detections.append(
                        tuple([file, "Pink_Blue", detected_signature, range_measurement, bearing_measurement]))
                else:
                    detected_signature = 1
                    landmark_location = tuple(
                        [x, y, w, 2 * h, detected_signature])
                    range_measurement = estimate_range(landmark_location)
                    bearing_measurement = estimate_bearing(img, landmark_location,
                                                           range_measurement)
                    detections.append(
                        tuple([file, "Blue_Pink", detected_signature, range_measurement, bearing_measurement]))

                show_data(range_measurement, bearing_measurement)

            if (LOWER_YELLOW[0] <= avg_hsv[0] <= UPPER_YELLOW[0] and
                LOWER_YELLOW[1] <= avg_hsv[1] <= UPPER_YELLOW[1] and
                    LOWER_YELLOW[2] <= avg_hsv[2] <= UPPER_YELLOW[2]):

                if (orientation == 1):
                    detected_signature = 4
                    landmark_location = tuple(
                        [x, y - h, w, 2 * h, detected_signature])
                    range_measurement = estimate_range(landmark_location)
                    bearing_measurement = estimate_bearing(img, landmark_location,
                                                           range_measurement)
                    detections.append(
                        tuple([file, "Pink_Yellow", detected_signature, range_measurement, bearing_measurement]))

                else:
                    detected_signature = 3
                    landmark_location = tuple(
                        [x, y, w, 2 * h, detected_signature])
                    range_measurement = estimate_range(landmark_location)
                    bearing_measurement = estimate_bearing(img, landmark_location,
                                                           range_measurement)
                    detections.append(
                        tuple([file, "Yellow_Pink", detected_signature, range_measurement, bearing_measurement]))

                show_data(range_measurement, bearing_measurement)


write_file('./measurement.dat', detections)
print("--This run took %0.2f seconds--" % (time.time() - startTime))
