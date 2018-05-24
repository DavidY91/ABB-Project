# -*- coding: utf-8 -*-
"""
This script finds the correspondences for two images.
The key method used here is implemented in
Myronenko A, Song X. Point set registration: coherent point drift[J]. IEEE Transactions on Pattern Analysis & Machine Intelligence, 2010, 32(12):2262-2275.
Author: Xu Yang
Last Modified: 2018/05/22
"""
import matlab.engine
import numpy as np
import cv2
import matplotlib.pyplot as plt
from feature_detection import feature_detection


def show_label(image, point2f, index):
    count = 0
    for point in point2f:
        image = cv2.putText(image, str(index[count]), tuple(point), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
        count += 1
    return image


# setting path
img_name_std = 'D:\\Documents\\ABB\\Data\\0510\\left\\4.bmp'
img_name_new = 'D:\\Documents\\ABB\\Data\\0510\\left\\7.bmp'
# visualize the matching result or not
visualization = True
# read images
img_std = cv2.imread(img_name_std, cv2.IMREAD_GRAYSCALE)
img_new = cv2.imread(img_name_new, cv2.IMREAD_GRAYSCALE)
# detect features
keypoints_std = feature_detection(img_std)
keypoints_new = feature_detection(img_new)
point2f_std = cv2.KeyPoint_convert(keypoints_std)
point2f_new = cv2.KeyPoint_convert(keypoints_new)
# start MATLAB engine
eng = matlab.engine.start_matlab()  # this is the most time consuming part, consider kick it out of your loop
# perform feature matching
x = point2f_new.tolist()
y = point2f_std.tolist()
reg = eng.calledByPython(x, y)
new2std_index = (np.array(reg['C']) - 1)[0]  # Python indexes from 0 while MATLAB indexes from 1
point2f_new_as_std = point2f_new[new2std_index, :]
# visualize the registration result (optional)
if visualization:
    im_std_with_keypoints = cv2.drawKeypoints(img_std, keypoints_std, np.array([]), (0, 0, 255))
    im_std_with_label = show_label(im_std_with_keypoints, point2f_std, np.array(range(point2f_std.shape[0])))
    im_new_with_keypoints = cv2.drawKeypoints(img_new, keypoints_new, np.array([]), (0, 0, 255))
    im_new_after_sorting = show_label(im_new_with_keypoints, point2f_new_as_std, np.array(range(point2f_new_as_std.shape[0])))
    plt.subplot(121)
    plt.imshow(im_std_with_label)
    plt.subplot(122)
    plt.imshow(im_new_after_sorting)
    plt.show()
