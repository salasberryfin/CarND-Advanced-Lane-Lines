import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip
import IPython

import img_operation
import camera_calibration

images_folder = "./test_images"
images_exp = "test*.jpg"
image_path = os.path.join(images_folder, images_exp )
images_list = glob.glob(image_path)

straight_exp = "straight*.jpg"
straight_path = os.path.join(images_folder, straight_exp )
straight_list = glob.glob(straight_path)


if __name__ == "__main__":
    white_output = 'test_videos_output/project-video-output.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(img_operation.undistort_images)
    get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')

