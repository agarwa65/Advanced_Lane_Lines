##Refernce: https://github.com/udacity/CarND-Camera-Calibration/blob/master/camera_calibration.ipynb

import numpy as np
import cv2
import glob
import math
import matplotlib.pyplot as plt

#prepare object points amd image points

def calc_caliberation_matrix(path):
    nx = 9
    ny = 6
    objp = np.zeros((nx*ny,3),np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(9,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    #path = 'camera_cal/calibration*.jpg'
    images = glob.glob(path)
    num_of_images = len(images)

    fig = plt.figure()
    i = 1
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny) , None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            ax = fig.add_subplot(math.ceil(num_of_images /2),2, i)
            chessboard = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            ax.imshow(chessboard)
            ax.axis('off')
            i +=1
    # Do camera calibration given object points and image points
    return cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None), fig



""" Undistort an image"""
def img_undistort(img, mtx, dist):
    """camera undistort image"""
    return cv2.undistort(img, mtx, dist)

def img_warped(img):
    img_size = [1280,720]
    l_offset, r_offset = 300, 1000

    # Define 4 source points src = np.float32([[,],[,],[,],[,]])
    # Define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    src_points = np.float32([[254,688],[592,451],[684,451],[1056,688]])
    #dst = np.float32([[250,720],[250,0],[1000,0],[1000,720])
    dst_points = np.float32([[l_offset,img_size[1]],[l_offset,0],[r_offset,0],[r_offset,img_size[1]]])
    # Get the transform matrix, M
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
    # Warp the image to top-down view
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    # Return the resulting image and matrix
    return warped, M, M_inv