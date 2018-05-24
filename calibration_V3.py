"""
Camera Calibration Program for 6 DOF Measurement Project

Author: Zhongyi Zhou
ABB
Apr. 24, 2018
"""
import numpy as np
import cv2
import glob
import time
import os
import shutil


# save 5 parameters from calibration
def save5para(ret, mtx, dist, rvecs, tvecs,savedir):
    if os.path.exists(savedir+"dist_para/") == 0:
        os.makedirs(savedir+"dist_para/")
    np.savez(savedir+"dist_para/5para.npz", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


def rename_images(dirname,fmt):
    """
    Rename the images filename from 1, 2 to 01, 02
    :param dirname: directory of the images
    :param fmt: format of the images. For example, ".bmp" pr ".jpg"
    :return: None
    """
    imgset = glob.glob(dirname+"*"+fmt)
    for fname in imgset:
        ind_line = fname.rindex("\\")
        ind_dot = fname.rindex(".")
        # print(ind_dot - ind_line)
        if ind_dot - ind_line == 3:
            continue
        else:
            if ind_dot - ind_line == 2:
                shutil.move(fname, fname[0:ind_line+1]+"0"+fname[ind_line+1:])
            else:
                print("image name format is invalid!")


# create and save images
def saveimage(dirname,filename,img):
    if os.path.exists(dirname) == 0:
        os.makedirs(dirname)
    cv2.imwrite(dirname+filename+".bmp",img)


# test one image from the testing directory
def dist_test(testdir, parafile):
    """
    Test the parameter from  the ".npz" parameter file.
    This function will create a new directory under testdir named "testfile".
    All the undistorted images will be saved there
    :param testdir: test data directory path
    :param parafile: internal parameter npz file
    :return: None
    """
    #imgflie = glob.glob(testdir + '*' + '.bmp')
    para = np.load(parafile)
    mtx = para['mtx']
    dist = para['dist']
    test_images = glob.glob(testdir + '*' + ".bmp")
    test_images.sort()
    for filename in test_images:
        #dist_test(testdir, imgfile, mtx, dist)

        #print(testdir,filename)
        img = cv2.imread(filename)
        # img = cv2.resize(img,(1296,972))
        h,  w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        #print("roi=",roi)
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x,y,w,h = roi
        #dst = dst[y:y+h, x:x+w]
        if os.path.exists(testdir+"testfile/") == 0:
            os.makedirs(testdir + "testfile/")

        cv2.imwrite(testdir+"testfile/"+filename[len(testdir):],dst)
        print("test file", filename," saving success!")


def show(img):
    """
    show the image
    :param img: image variable
    :return: None
    """
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600,600)
    cv2.imshow('image',img)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('messigray.png',img)
        cv2.destroyAllWindows()

# train 5 parameters from training the image directory
# image directory example :"/home/" 
# "/" at last is necessary
# image format example: ".bmp"
# "." at first is necessary


def multi_imgpt(imgdir, imgfmt, ret_r = 5, ret_c = 6):
    """
    This function is used to recognize the image points (in pixel dimension)
    on the chess board.

    :param imgdir: directory which contains target images
    :param imgfmt: image format
    :param ret_r: row number of the rectangle
    :param ret_c: column number of the rectangle
    :return: a bool list indicating whether the chessboard has been successfully
    recognized and a list of recognized image points
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.elo
    ret_list = []
    # rename the image from 1.bmp to 01.bmp
    rename_images(imgdir, imgfmt)
    # makedir to visualize the recognition point suquence.
    if os.path.exists(imgdir+"cdn_success/") == 1:
        shutil.rmtree(imgdir+"cdn_success/")
    os.makedirs(imgdir + "cdn_success/")

    images = glob.glob(imgdir + '*' + imgfmt)
    images.sort()
    print("images =", images)
    totalnum = images.__len__()
    counter = 0
    loop_ctr = 1
    # start recognize the imgpt
    for fname in images:
        ind_line = fname.rindex('/')
        ind_dot = fname.rindex('.')
        imagename = fname[ind_line + 1:]
        print("image", imagename, " start!")
        # print("image",loop_ctr, "start!")
        img = cv2.imread(fname)
        #img = cv2.resize(img,(1296,972))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        objp = np.zeros((ret_r*ret_c, 3), np.float32)
        objp[:,:2] = np.mgrid[0:ret_r, 0:ret_c].T.reshape(-1, 2)
        ret, corners = cv2.findChessboardCorners(gray, (ret_r, ret_c), None)
        ret_list.append(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            print("image", imagename, "success!")
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2.reshape(-1,2))
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (ret_c, ret_r), corners2, ret)
            #print(corners2.shape)
            cv2.circle(img, (int(corners2[0,0,0]), int(corners2[0,0,1])), 10, (255, 255, 255), -11)
            counter += 1
            print("image", fname[ind_line + 1:], "successed!")
            saveimage(imgdir + "cdn_success/", fname[ind_line + 1:ind_dot], img)
        else:
            print("image", loop_ctr, "failure!")
            imgpoints.append(None)
        cv2.destroyAllWindows()
        loop_ctr += 1
    return ret_list, imgpoints


def dist_train(imgdir, imgfmt, ret_r = 5, ret_c = 6):
    """
    This function is used to train the internal coefficient through the calibration images
    in the imgdir.

    :param imgdir: calibration images dictonary path
    :param imgfmt: calibration images format
    :param ret_r: the row number of the rectangle
    :param ret_c: the column number of the rectangle
    :return: retvalue, internal matrix, distortion coefficient array, rvecs, tvecs, sucess filename list
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    rename_images(imgdir, imgfmt)
    #images = glob.glob('./calibration/5/*.bmp')
    images = glob.glob(imgdir+'*'+imgfmt)
    images.sort()
    print("images =", images)
    counter = 0
    loop_ctr = 1

    for fname in images:
        ind_line = fname.rindex('\\')
        ind_dot = fname.rindex('.')
        print("image", fname[ind_line+1:], " start!")
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        objp = np.zeros((ret_r*ret_c,3), np.float32)
        objp[:,:2] = np.mgrid[0:ret_r,0:ret_c].T.reshape(-1,2)
        ret, corners = cv2.findChessboardCorners(gray, (ret_r,ret_c),None)

        # If found, add object points, image points (after refining them)
        sucess_fname_list = []
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            #print("corners=",corners)
            #print("corners2 = corners???",corners2==corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (ret_r,ret_c), corners2,ret)

            counter += 1
            # print("fname = ",fname)


            print("image",fname[ind_line:],"successed!")
            sucess_fname_list.append(fname[ind_line+1:ind_dot])

            saveimage(fname[0:ind_line+1]+"success/", fname[ind_line+1:ind_dot],img)

        else:
            print("image", fname[ind_line:], "failed!")

            saveimage(fname[0:ind_line + 1] + "fail/", fname[ind_line + 1:ind_dot], img)
            #print("image",loop_ctr,"failed!")
            #saveimage(imgdir+"fail/",str(loop_ctr),img)
        cv2.destroyAllWindows()
        loop_ctr += 1
    print("success number=",counter)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    print("ret =",ret)
    t_train_1 = time.time()
    #print("training cost=", t_train_1 - t_train_0, 's')
    return ret, mtx, dist, rvecs, tvecs, sucess_fname_list


def dist_main(traindir, testdir = None):
    t0 = time.time()

    #[ret, mtx, dist, rvecs, tvecs] = dist_train('./calibration/5/','.bmp')
    if os.path.exists(traindir+"success/") == 1:
        shutil.rmtree(traindir+"success/")
    if os.path.exists(traindir + "fail/") == 1:
        shutil.rmtree(traindir + "fail/")
    if os.path.exists(traindir + "testfile/") == 1:
        shutil.rmtree(traindir + "testfile/")
    [ret, mtx, dist, rvecs, tvecs, success_fname_list] = dist_train(traindir, '.bmp')
    save5para(ret, mtx, dist, rvecs, tvecs, traindir)
    print("success_fname_list =",success_fname_list)
    t1 = time.time()
    print("calibration time cost =", t1 - t0, "s")
    if testdir != None:
        #test_images = glob.glob(testdir + '*' + ".bmp")
        parafile = traindir + "dist_para/5para.npz"
        dist_test(testdir, parafile)
    return ret, mtx, dist, rvecs, tvecs


if __name__ == '__main__':
    left_clb = "D:/Documents/ABB/Data/0510/calib_left/"
    right_clb = "D:/Documents/ABB/Data/0510/calib_right/"
    dist_main(left_clb, left_clb)
    dist_main(right_clb, right_clb)

    # dist_main("./double_clb_0301/right/", "./double_clb_0301/right/")
    #dist_main("./4clb/4/", "./cdn_trans/4/")
    #dist_main("./4clb/5/", "./cdn_trans/5/")




