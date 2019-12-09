import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

images_folder = "./camera_cal"
images_exp = "calibration*.jpg"
images_path = os.path.join(images_folder, images_exp )

images = glob.glob(images_path)

output_folder = "./output_images/chessboard_calibration"

nx = 9
ny = 6

def find_corners(gray, n):
    objpoints = []
    imgpoints = []
    objp = np.zeros((n[0]*n[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:n[0], 0:n[1]].T.reshape(-1, 2)
    ret, corners = cv2.findChessboardCorners(gray, (n[0], n[1]), None)
    if ret is True:
        imgpoints.append(corners)
        objpoints.append(objp)
        
        return imgpoints, objpoints

    return None


def calibrate_undistort(image, imgpoints, objpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       image.shape[::-1],
                                                       None,
                                                       None)
    dst = cv2.undistort(image,
                        mtx,
                        dist,
                        None,
                        mtx)

    return dst, mtx, dist
    

def draw_undist_corners(img, n):
    ret, corners = cv2.findChessboardCorners(img, n, None)
    src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
    if ret is True:
        cv2.drawChessboardCorners(img, n, corners, ret)

        return img, src


def transform_perspective(img, src, offset=100):
    img_size = (img.shape[1], img.shape[0])
    dest = np.float32([[180, 250], [180, 720], [1000, 250], [1000, 720]])
    M = cv2.getPerspectiveTransform(src, dest)
    inv = cv2.getPerspectiveTransform(dest, src)
    warped = cv2.warpPerspective(img, M, (img_size[0], img_size[1]), flags=cv2.INTER_NEAREST)

    return warped, M, inv


if __name__ == '__main__':
    for item in images:
        print("Image: ", item)
        current_name = item.split("/")[2]
        image = cv2.imread(item)
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            imgpoints, objpoints = find_corners(gray, (nx, ny))
            dst = calibrate_undistort(image, imgpoints, objpoints)
            undist_corners, src = draw_undist_corners(dst, (nx, ny))
            warped, M = transform_perspective(undist_corners, src)
            output_path = os.path.join(output_folder, 
                                       "warped-{}".format(current_name))
            print("Matrix: ", M)
            cv2.imwrite(output_path, warped)
        except:
            print("Something went wrong with {0}: {1}", current_name)
        # plt.imshow(warped)
        # plt.show()

