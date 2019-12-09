import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import camera_calibration

output_folder = "./output_images"

calibration_folder = "./camera_cal"
calibration_image = "calibration3.jpg"
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


class Lane():
    """
        The Lane class holds all relevant information about lane detection.
    """
    def __init__(self, ploty, left_fit, right_fit):
        self.ploty = ploty
        self.left_fit = left_fit
        self.right_fit = right_fit
        self.left_curv_real = None
        self.left_curv_pix = None
        self.right_curv_real = None
        self.right_curv_pix = None
        self.center_real = None
        self.center_pix = None
        self.left_bottom_real = None
        self.left_bottom_pix = None
        self.right_bottom_real = None
        self.right_bottom_pix = None
        self.left_fitx = None
        self.right_fitx = None


def identify_lanes(binary_warped):
    """
        Identify lanes lines from binary image.
    """
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        # TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    """
        Fit second order polynomial and generate values for plotting.
    """
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = identify_lanes(binary_warped)

    # TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    lane = Lane(ploty, left_fit, right_fit)
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        lane.left_fitx = left_fitx
        lane.right_fitx = right_fitx
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()

    return out_img, lane


def measure_curvature(binary_warped, lane):
    """
        Calculate the lines' curvature in meters.
    """
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700
    ploty = lane.ploty
    left_fit = lane.left_fit
    right_fit = lane.right_fit
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    left_fit_cr = np.polyfit(ym_per_pix*ploty, xm_per_pix*leftx, 2)
    right_fit_cr = np.polyfit(ym_per_pix*ploty, xm_per_pix*rightx, 2)
    y_eval = np.max(ploty)
    
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    left_lane_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
    right_lane_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]
    lane_center = (left_lane_bottom + right_lane_bottom)/2.
    center_image = 640
    center = (lane_center - center_image)*xm_per_pix
    position = "left" if center < 0 else "right"
    center = "Vehicle is {:.2f}m {} of center".format(center, position)

    lane.left_curv_real = left_curverad
    lane.right_curv_real = right_curverad
    lane.center_real = center
    lane.left_bottom_real = left_lane_bottom
    lane.right_bottom_real = right_lane_bottom
    
    return lane


def measure_curvature_pixels(lane):
    """
        Calculate the lines' curvature in pixels.
    """
    ploty = lane.ploty
    right_fit = lane.right_fit
    left_fit = lane.left_fit
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    left_lane_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
    right_lane_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]

    lane.left_curv_pix = left_curverad
    lane.right_curv_pi = right_curverad
    lane.left_bottom_pix = left_lane_bottom
    lane.right_bottom_pix = right_lane_bottom

    return lane


def draw_output(image, warped, inv, lane):
    """
        Draw the detected lane space over the original undistorted image
    """
    new_img = np.copy(image)
    prep_img = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((prep_img, prep_img, prep_img))
    warped_size = (warped.shape[1], warped.shape[0])
    ploty = lane.ploty
    left_fitx = lane.left_fit[0]*ploty**2 + lane.left_fit[1]*ploty + lane.left_fit[2]
    right_fitx = lane.right_fit[0]*ploty**2 + lane.right_fit[1]*ploty + lane.right_fit[2]
    points_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    points_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    points = np.hstack((points_left, points_right))
    cv2.fillPoly(color_warp, np.int_([points]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([points_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([points_right]), isClosed=False, color=(0,255,255), thickness=15)
    newwarp = cv2.warpPerspective(color_warp, inv, (warped_size[0], warped_size[1])) 
    lane_image = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)

    return lane_image


def insert_data(image, lane):
    text_image = np.copy(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    curv_position = (10, 30)
    center_position = (10, 70)
    scale = 1
    color = (0, 255, 0)
    line = 2

    left_curvature = "Left curvature: %.2fm" % lane.left_curv_real
    cv2.putText(text_image, left_curvature, 
                curv_position, 
                font, 
                scale,
                color,
                line)
    cv2.putText(text_image, lane.center_real, 
                center_position, 
                font, 
                scale,
                color,
                line)

    return text_image


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
    dst, mtx, dist = camera_calibration.calibrate_undistort(gray, imgpoints, objpoints)
    undist_corners, src = camera_calibration.draw_undist_corners(dst, (nx, ny))

    return src, mtx, dist


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
    output_path = "output_images/new_undist/undist-{}".format(img_name)
    cv2.imwrite(output_path, output_img)


def undistort_images(img):
    """
        Returns undistorted image.
    """
    src, mtx, dist = get_calibration()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = cv2.undistort(img, mtx, dist, None, mtx)
    out_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    binary_gradient = gradient_magnitude(out, (30, 100))
    binary_hls = hls_thresh(apply_hls(out), (120, 255))
    abs_sobelx, abs_sobely = sobel(gray, 3)
    binary_dir = dir_threshold(gray, abs_sobelx, abs_sobely, sobel_kernel=15, thresh=(0.7, 1.3))
    frame = UndistImage(out,
                        out_gray,
                        binary_hls,
                        binary_gradient,
                        binary_dir)
    combine = np.zeros_like(frame.gradient)
    combine[(frame.gradient == 1) | ((frame.hls == 1) & (frame.direction == 1))] = 1
    leftupperpoint  = [570, 470]
    rightupperpoint = [720, 470]
    leftlowerpoint  = [270, frame.hls.shape[0]]
    rightlowerpoint = [1050, frame.hls.shape[0]]
    trans_mat = np.float32([leftupperpoint, leftlowerpoint, rightupperpoint, rightlowerpoint])
    warped, M, inv = camera_calibration.transform_perspective(combine*255, trans_mat)
    result_img, lane = fit_polynomial(warped)
    lane = measure_curvature(warped, lane)
    lane = measure_curvature_pixels(lane)
    ex_result = draw_output(frame.image, warped, inv, lane)
    text_image = insert_data(ex_result, lane)

    return text_image
