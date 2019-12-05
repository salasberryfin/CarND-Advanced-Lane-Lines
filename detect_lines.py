import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

import undistort_img

images_folder = "./test_images"
images_exp = "test*.jpg"
image_path = os.path.join(images_folder, images_exp )
images_list = glob.glob(image_path)


if __name__ == "__main__":
    undistorted_imgs = undistort_img.undistort_images(images_list)
