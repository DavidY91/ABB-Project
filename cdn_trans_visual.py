"""
Coordinate Transformation Visualization


Author: Zhongyi Zhou
ABB
Apr. 24, 2018

"""

import numpy as np
from mayavi import mlab
import xlrd


def read_sheet(filename):
    """
    This function is used to read the excel file.
    The recording format should be as follow:
    No. X  Y  Z  alpha  beta  gamma
    1
    2
    3
    4
    5
    6
    .
    .
    .

    So the function will read the data from B1 to the end.

    :param filename: excel filepath
    :return: measured data
    """
    data = xlrd.open_workbook(filename)
    table = data.sheets()[0]
    DOF = table.ncols - 1
    sample_num = table.nrows - 1
    mea_data = np.zeros((sample_num, DOF))
    # print("sample number =", sample_num)
    # print("degree of freedom =", DOF)
    for i in range(1, table.nrows):
        for j in range(1, table.ncols):
            # print(table.cell(i, j).value)
            mea_data[i - 1, j - 1] = table.cell(i, j).value

    return mea_data


def axis_plot(kp_arr):
    """
    Plot the axiss according to 4 key points.
    RGB -> XYZ

    :param kp_arr: key points array
    :return: None
    """
    if kp_arr.shape != (4,4):
        print("kp_arr format error!")
    else:
        mlab.plot3d([kp_arr[0,0],kp_arr[0,1]], [kp_arr[1,0],kp_arr[1,1]], [kp_arr[2,0],kp_arr[2,1]], color = (1,0,0), tube_radius = 1)
        mlab.plot3d([kp_arr[0,0],kp_arr[0,2]], [kp_arr[1,0],kp_arr[1,2]], [kp_arr[2,0],kp_arr[2,2]], color = (0,1,0), tube_radius = 1)
        mlab.plot3d([kp_arr[0,0],kp_arr[0,3]], [kp_arr[1,0],kp_arr[1,3]], [kp_arr[2,0],kp_arr[2,3]], color = (0,0,1), tube_radius = 1)

def axis_plot_all(axis_list):
    """
    Plot all the coordinate axis in the axis_list
    RGB -> XYZ

    :param axis_list: list of key points arrays
    :return: None
    """
    for arr in axis_list:
        axis_plot(arr)
    mlab.show()

def visual_step_by_step(homo_model, axis_base_arr, test_ind):
    """
    This function helps beginners do coordinate transformation and visualize it step by step.

    :param homo_model: dictionary data loaded from .npz file, containing the transformation data
    :param axis_base_arr: the base axis array
    :param test_ind: indicator of test image.
    :return: None
    """
    objbase2objstd_list = homo_model['objbase2objstd_list']
    objstd2obj_list = homo_model['objstd2obj_list']
    obj2cam_list = homo_model['obj2cam_list']
    cam2board = homo_model['cam2board'] 
    board2std = homo_model['board2std']
    std2base = homo_model['std2base']
    print("data loading success!")

    # base
    axis_base = axis_base_arr
    axis_list.append(axis_base)
    print("axis_base =", axis_base)
    axis_plot_all(axis_list)

    # std
    axis_std = np.dot(std2base, axis_base_arr)
    axis_list.append(axis_std)
    print("axis_std =", axis_std)
    axis_plot_all(axis_list)

    # board
    axis_board = std2base @ board2std @ axis_base_arr
    axis_list.append(axis_board)
    print("axis_board =", axis_board)
    axis_plot_all(axis_list)

    #camera
    #axis_cam = np.dot(cam2board, axis_board)
    axis_cam = std2base @ board2std @ cam2board @ axis_base_arr
    axis_list.append(axis_cam)
    print("axis_cam =", axis_cam)
    axis_plot_all(axis_list)

    #object
    obj2cam = obj2cam_list[test_ind]
    #axis_obj = obj2cam @ axis_cam
    axis_obj = std2base @ board2std @ cam2board @ obj2cam @ axis_base_arr
    axis_list.append(axis_obj)
    print("axis_obj =", axis_obj)
    axis_plot_all(axis_list)

    #base2
    objstd2obj = objstd2obj_list[test_ind]
    axis_objstd = std2base @ board2std @ cam2board @ obj2cam @ objstd2obj @ axis_base_arr
    axis_list.append(axis_objstd)
    print("axis_objstd =", axis_objstd)
    axis_plot_all(axis_list)

    objbase2objstd = objbase2objstd_list[test_ind]
    axis_objbase = std2base @ board2std @ cam2board @ obj2cam @ objstd2obj @ objbase2objstd @ axis_base_arr
    axis_list.append(axis_objbase)
    print("axis_objbase =", axis_objbase)
    axis_plot_all(axis_list)


if __name__=='__main__':
    wld_excel_path = "D:/Documents/ABB/Data/0416/baseImg/gtruth.xlsx"

# base
    axis_base = [[0.0, 0.0, 0.0, 1.0],[50.0, 0.0, 0.0, 1.0],[0.0, 50.0, 0.0, 1.0],[0.0, 0.0, 50.0, 1.0]]
    axis_list = []
    axis_base_arr = np.array(axis_base).T
    print("axis_base_arr =", axis_base_arr)
    axis_list.append(axis_base_arr)
    print("first trans display finished!")

# read data
    cam_fig_path = "D:/Documents/ABB/Data/0416/calibration/downleft/temp"
    datafile = cam_fig_path + "/homo_trans.npz"
    homo_model = np.load(datafile)
    objbase2objstd_list = homo_model['objbase2objstd_list']
    objstd2obj_list = homo_model['objstd2obj_list']
    obj2cam_list = homo_model['obj2cam_list']
    cam2board = homo_model['cam2board']
    board2std = homo_model['board2std']
    std2base = homo_model['std2base']
    print("data loading success!")

    trials = 5
    DOF6 = read_sheet(wld_excel_path)
    test_ind = trials
    visual_step_by_step(homo_model, axis_base_arr, test_ind)
