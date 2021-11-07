import numpy as np
import cv2
import glob
import pickle


################ CAMERA CALIBRATION WITH A CHESSBOARD - OBJECT POINTS AND IMAGE POINTS DETECTION ######

## If your image window needs to be resized to show correctly in your screen
#cv2.namedWindow("myImage", cv2.WINDOW_NORMAL)
#cv2.resizeWindow('myImage', 1280, 720)

# Termination criteria. We stop either when an accuracy is reached or when
# # we have finished the number of iterations
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

## Change according to your inputted chessboard
numX_inside_squares = 7
numY_inside_squares = 7
checkerboard = (numX_inside_squares, numY_inside_squares)

objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

## If you want to have the output in mm
# size_of_chessboard_squares_mm = 20
# objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
obj_points = [] # 3d point in real world space
img_points = [] # 2d points in image plane.

path_file = '..' + '*.jpeg'
images = glob.glob(path_file)

found = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)

    # If corners are found in the image
    if ret == True:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1,-1), criteria)
        img_points.append(corners)
        # Draw the corners and display them
        cv2.drawChessboardCorners(img, checkerboard, corners2, ret)
        cv2.imshow("myImage", img)
        found += 1
        ## Save all the images where corners are detected in a folder called input
        cv2.imwrite('{}'.format(fname.replace('input', 'corners', 1)), img)
        cv2.waitKey(30)

print("Number of images with corners detected: ", found)

cv2.destroyAllWindows()



############## CALIBRATION #######################################################
## Use all the images with corners detected to calibrate our camera
ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

############## UNDISTORTION #####################################################
## Get one image to apply undistortion
distorced_img = cv2.imread('...jpeg')
height, width = distorced_img.shape[:2]
optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (width, height), 1, (width, height))

## undistort
undistorced_img = cv2.undistort(distorced_img, camera_matrix, dist, None, optimal_camera_matrix)

## crop the image
x, y, w, h = roi
undistorced_img = undistorced_img[y:y+h, x:x+w]
cv2.imwrite('../calib_result_cropping.png', undistorced_img)



## Calculate the reprojection error
mean_error = 0
for i in range(len(obj_points)):
    img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist)
    error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2)/len(img_points2)
    mean_error += error
print( "total error: {}".format(mean_error/len(obj_points)) )


## Save the camera calibration results in a pickle
calib_result = {}
calib_result["mtx"] = camera_matrix
calib_result["optimal_camera_matrix"] = optimal_camera_matrix
calib_result["dist"] = dist
calib_result["rvecs"] = rvecs
calib_result["tvecs"] = tvecs
pickle.dump(calib_result, open("../camera_calib_results.p", "wb"))


## Apply the calibrated camera parameters to a input image or video frame (distorted_image)
# undistorced_image = cv2.undistort(distorced_image, mtx, dist, None,
#                                     optimal_camera_matrix)
