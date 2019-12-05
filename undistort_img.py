import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

import camera_calibration

calibration_folder = "./camera_cal"
calibration_image = "calibration17.jpg"
calibration_path = os.path.join(calibration_folder, calibration_image)

images_folder = "./test_images"
images_exp = "test*.jpg"
image_path = os.path.join(images_folder, images_exp )
images_list = glob.glob(image_path)

output_folder = "./output_images"

nx = 9
ny = 6


def prepare_image(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image, gray


def get_calibration():
    image, gray = prepare_image(calibration_path)
    imgpoints, objpoints = camera_calibration.find_corners(gray, (nx, ny))
    dst = camera_calibration.calibrate_undistort(gray, imgpoints, objpoints)
    undist_corners, src = camera_calibration.draw_undist_corners(dst, (nx, ny))

    return src


if __name__ == "__main__":
    src = get_calibration()
    # img, gray = prepare_image(image_path)
    for new_path in images_list:
        img_name = new_path.split("/")[2]
        img, gray = prepare_image(new_path)
        warped, M = camera_calibration.transform_perspective(img, src)
        output_path = "output_images/road_undistort/calibration-{}".format(img_name)
        cv2.imwrite(output_path, warped)
        import pdb; pdb.set_trace()
