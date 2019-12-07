import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

import img_operation

images_folder = "./test_images"
images_exp = "test*.jpg"
image_path = os.path.join(images_folder, images_exp )
images_list = glob.glob(image_path)


if __name__ == "__main__":
    undistorted_imgs = img_operation.undistort_images(images_list)
    for frame in undistorted_imgs:
        combine = np.zeros_like(frame.gradient)
        combine[(frame.gradient == 1) | ((frame.hls == 1) & (frame.direction == 1))] = 1
        # cv2.imwrite("output_images/binary_lane_detection/detection-{}.jpg".format(name), combine*255)
        # plt.imshow(combine, cmap="gray")
        # import pdb;pdb.set_trace()

