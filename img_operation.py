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

THRESH = (130, 255)


class UndistImage:
    """ 
        The UndistImage class holds all forms of the undistorted img
    """
    def __init__(self, image, gray, hls, gradient, direction):
        self.image = image
        self.gray = gray
        self.hls = hls
        self.gradient = gradient
        self.direction = direction


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


def hls_thresh(img, thresh):
    s = img[:, :, 2]
    binary_s = np.zeros_like(s)
    binary_s[(s > thresh[0]) & (s <= thresh[1])] = 1

    return binary_s


def apply_hls(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    return hls


def gradient_magnitude(img, thresh, orient='x'):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if 'x' in orient:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    if 'y' in orient:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary


def sobel(gray, sobel_kernel):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    return abs_sobelx, abs_sobely


def dir_threshold(gray, abs_sobelx, abs_sobely, sobel_kernel=3, thresh=(0, np.pi/2)):
    direction = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(direction)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1

    return binary_output


def save_image_example(new_path, output_img):
    img_name = new_path.split("/")[2]
    output_path = "output_images/road_undistort_dir/dir-{}".format(img_name)
    cv2.imwrite(output_path, output_img)


def undistort_images(images_list):
    """
        Returns a list of undistorted images from a given set of pictures.
    """
    output_images = []
    src = get_calibration()
    for new_path in images_list:
        img, gray = prepare_image(new_path)
        out, out_gray = camera_calibration.transform_perspective(img, src)
        binary_gradient = gradient_magnitude(out, (30, 100))
        binary_hls = hls_thresh(apply_hls(out), (140, 255))
        abs_sobelx, abs_sobely = sobel(gray, 3)
        binary_dir = dir_threshold(gray, abs_sobelx, abs_sobely, sobel_kernel=15, thresh=(0.7, 1.3))
        image_atts = UndistImage(out,
                                 out_gray,
                                 binary_hls,
                                 binary_gradient,
                                 binary_dir)
        output_images.append(image_atts)
        # save_image_example(new_path, binary_dir*255)

    return output_images
