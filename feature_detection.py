"""
This function detects the circular features on a car model
Author: Xu Yang
Last Modified: 2018/05/22
"""


import cv2


def feature_detection(image):
    Param_bright = cv2.SimpleBlobDetector_Params()
    Param_dark = cv2.SimpleBlobDetector_Params()

    # parameter settings
    Param_bright.filterByColor = Param_dark.filterByColor = True
    Param_bright.blobColor = 255
    Param_dark.blobColor = 0

    Param_bright.minThreshold = Param_dark.minThreshold = 50
    Param_bright.maxThreshold = Param_dark.maxThreshold = 255
    Param_bright.thresholdStep = Param_dark.thresholdStep = 2

    Param_bright.filterByArea = Param_dark.filterByArea = True
    Param_bright.minArea = Param_dark.minArea = 30
    Param_bright.maxArea = Param_dark.maxArea = 1000

    Param_bright.filterByCircularity = Param_dark.filterByCircularity = True
    Param_bright.minCircularity = Param_dark.minCircularity = 0.7

    # construct detectors for dark blob and bright blob
    detector_bright = cv2.SimpleBlobDetector_create(Param_bright)
    detector_dark = cv2.SimpleBlobDetector_create(Param_dark)

    keypoints_dark = detector_dark.detect(image)
    keypoints_bright = detector_bright.detect(image)
    keypoints = keypoints_bright + keypoints_dark
    return keypoints