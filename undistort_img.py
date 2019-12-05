import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import camera_calibration

output_folder = "./output_images"

calibration_folder = "./camera_cal"
calibration_image = "calibration17.jpg"
calibration_path = os.path.join(calibration_folder, calibration_image)


def prepare_image(path):
    """
        Generic method that reads an image from a given path.
        Returns the original image and its grayscale version
    """
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image, gray


def get_calibration():
    """
        Calibrates the camera from one of the chessboard images 
        and returns the correction coefficient that needs to be applied
        to undistort a new picture.
    """
    nx = 9
    ny = 6
    image, gray = prepare_image(calibration_path)
    imgpoints, objpoints = camera_calibration.find_corners(gray, (nx, ny))
    dst = camera_calibration.calibrate_undistort(gray, imgpoints, objpoints)
    undist_corners, src = camera_calibration.draw_undist_corners(dst, (nx, ny))

    return src


def save_image_example(new_path, output_img):
    img_name = new_path.split("/")[2]
    output_path = "output_images/road_undistort/calibration-{}".format(img_name)
    cv2.imwrite(output_path, output_img)


def undistort_images(images_list):
    """
        Returns a list of undistorted images from a given set of pictures.
    """
    output_images = []
    src = get_calibration()
    for new_path in images_list:
        img, gray = prepare_image(new_path)
        output_img, M = camera_calibration.transform_perspective(img, src)
        output_images.append(output_img)
        # save_image_example(new_path, output_img)

    return output_images
