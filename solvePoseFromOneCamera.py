"""
Solve pose from one camera using sufficient 2D-3D pair.
Author: Xu Yang

==============
Another implementation from a former intern Zhongyi Zhou could be found here:
https://github.com/JoeyChuuichi/ABB_MachineVisionIntern/blob/master/solvePnP.py
The overall transformation was formulated in line 176: "final_homo = std2base @ board2std @ cam2board @ obj2cam @ objstd2obj @ objbase2objstd" .
I've not got a chance to test its correctness personally, however, it looks that some of transformation could be merged.

In this version, three coordinate frames are used in the following script:
object coordinate frame: fixed on the object (car model)
camera coordinate frame: derived from intrinsic calibration.
base coordinate frame: the origin is defined at the center of the base(云台）. XOY plane is its top plane.

This script uses the "homogene()" function defined in solvePnP.py.
"""
import numpy as np
import cv2 as cv
from feature_matching import point2f_std, point2f_new_as_std
from solvePnP import homogene
from extract3Dcoordinates import coordinates as object3D


def findValid3D2DPair(point3d, point2d):
    mask = np.ones(point3d.shape[0], dtype=bool)
    for i in range(point3d.shape[0]):
        if point3d[i, 0] == point3d[i, 1] == point3d[i, 2] == 0 or point2d[i, 0] == point2d[i, 1] == 0:
            mask[i] = False
    return point3d[mask], point2d[mask]


# Load intrinsic parameter
camera_int_clb = "D:/Documents/ABB/Data/0510/calib_left/dist_para/5para.npz"
with np.load(camera_int_clb) as X:
    cameraMatrix, distCoeffs, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
translation = np.array([[147.93, 93.58, -15]])  # the translation vector from the object coordinate frame to
# the base coordinate frame. So, it changes as the object coordinate frame you define
object3D_std, imagePoints_std = findValid3D2DPair(object3D, point2f_std)
object3D_std = object3D_std - translation
object3D_new, imagePoints_new = findValid3D2DPair(object3D, point2f_new_as_std)
object3D_new = object3D_new - translation
_, R1, T1, inliers = cv.solvePnPRansac(object3D_std, imagePoints_std, cameraMatrix, distCoeffs, reprojectionError=3.0)
TransStd2Cam = homogene(R1, T1)
_, R2, T2, inliers2 = cv.solvePnPRansac(object3D_new, imagePoints_new, cameraMatrix, distCoeffs, reprojectionError=3.0)
TransNew2Cam = homogene(R2, T2)
new2std = np.dot(np.linalg.inv(TransStd2Cam), TransNew2Cam)  # transformation takes object from new location to the previous one

# TODO: compare the transformation (new2std) with the measurements for evaluating accuracy




