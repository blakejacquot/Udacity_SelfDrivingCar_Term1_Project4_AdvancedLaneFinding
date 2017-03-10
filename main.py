"""
    Docstring
"""

# Import standard libraries
import glob
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import statistics

from moviepy.editor import VideoFileClip
import cv2


def compute_camera_cal(dir_cal_images):
    """Compute distortion parameters for chessboard jpg's in a directory

    Args:
        dir_cal_images: Directory of calibration chessboard images

    Returns:
        objpoints: List of 3d points in real world space.
        imgpoints: List of 2d points in image plane.

    """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    num_x = 9 # Number of inside corners in x-direction
    num_y = 6 # Number of inside corners in y-direction
    objp = np.zeros((num_y*num_x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_x, 0:num_y].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    search_phrase = os.path.join('.', dir_cal_images, 'calibration*.jpg')
    images = glob.glob(search_phrase)

    # Step through the list and search for chessboard corners
    for fname in images:
        print('Processing ', fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (num_x, num_y), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (num_x, num_y), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()
    return objpoints, imgpoints

def undistort_image(objpoints, imgpoints, img):
    """Undistort an image with calibration parameters

    Args:
        objpoints: List of 3d points in real world space.
        imgpoints: List of 2d points in image plane.
        img: 2D image.

    Returns:
        undistorted_img:
    """
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
        gray.shape[::-1], None, None)
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted_img

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Args:
        img: 3 or 1 channel image
        vertices: np array of form np.float32([[top_left, top_rig, bot_rig, bot_left]])

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.

    For some reason doesn't work on 1-channel images
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    vertices = vertices.astype(int)

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(mask, img)
    return masked_image

def get_warp_params(img, src, dst):
    """
    Docstring
    """

    # Calculate perspective parameters for warping and unwarping
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv

def warper(img, M):
    """
    Docstring
    """
    img_shape = img.shape
    img_size = (img_shape[1],img_shape[0])
    img_proc = cv2.warpPerspective(img, M, img_size)
    return img_proc

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    Args:
      img: Color image on to draw lines.
      lines: numpy.ndarray of size (x, 1, 4). The line points x1,y1,x2,y2 are numpy.ndarray
        of size (4,)
      color: Color of superimposed lines.
      thickness: Thickness of lines (in pixels?).

    Returns:
      TBD

    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below

    """
    #line_image = np.copy(img)*0 # Blank image on which to draw lines.
    line_image = img
    shape_img = img.shape
    x_max = shape_img[1]
    for line in lines:
        for x1,y1,x2,y2 in line:
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    return line_image

def hough_lines(img, rho = 3, theta = np.pi/180, threshold = 10,
    min_line_len = 20, max_line_gap = 50):
    """
    `img` should be the output of a Canny transform.

    Returns
      line_img: Image with hough lines drawn.
      lines: Hough lines from the transform of form x1,y1,x2,y2.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
        minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

# Calculate directional gradient
def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply x or y gradient
    if orient == 'x':
    	sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
    	sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute values
    sobel = np.absolute(sobel)
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    # Return the result
    return binary_output

# Calculate gradient magnitude
def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1
    # Return the result
    return binary_output

# Calculate gradient direction
def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Error statement to ignore division and invalid errors
    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan(sobely/sobelx))
        dir_binary =  np.zeros_like(absgraddir)
        dir_binary[(absgraddir > thresh[0]) & (absgraddir < thresh[1])] = 1
    # Return the result
    return dir_binary


def make_binary_image(img):
    """ Process color image to make binary image with candidates for lane lines.
    Args:
      img: 3D color numpy array

    Returns:
      img_canny_blur_gray: 3D numpy array binary image with blur and Canny edge detection
        applied.
    """
    # Grayscale the image.
    gray_image = grayscale(img)

    # Gaussian blur the image
    ksize = 7
    img_blur_gray = gaussian_blur(gray_image, ksize)

    ksize = 3

    gradx = abs_sobel_thresh(img_blur_gray, orient='x', sobel_kernel=ksize, thresh=(10, 255))
    grady = abs_sobel_thresh(img_blur_gray, orient='y', sobel_kernel=ksize, thresh=(60, 255))
    mag_binary = mag_thresh(img_blur_gray, sobel_kernel=ksize, mag_thresh=(40, 255))
    dir_binary = dir_threshold(img_blur_gray, sobel_kernel=ksize, thresh=(.65, 1.05))

    # Combine all the thresholding information
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # Get hls channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #h = hls[:,:,0]
    #l = hls[:,:,1]
    s = hls[:,:,2]
    # Filter on s channel in coordination with the combined image
    s_binary = np.zeros_like(combined)
    s_binary[(s > 160) & (s < 255)] = 1
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors

    HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # For yellow
    yellow = cv2.inRange(HSV, (20, 100, 100), (50, 255, 255))
    # For white
    sensitivity_1 = 68
    white = cv2.inRange(HSV, (0,0,255-sensitivity_1), (255,20,255))
    sensitivity_2 = 60
    HSL = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    white_2 = cv2.inRange(HSL, (0,255-sensitivity_2,0), (255,255,sensitivity_2))
    white_3 = cv2.inRange(img, (200,200,200), (255,255,255))
    #bit_layer = combined | yellow | white | white_2 | white_3
    color_binary = np.zeros_like(combined)
    color_binary[(s_binary > 0) | (combined > 0) | (yellow > 0) | (white > 0) | (white_2 > 0) | (white_3 > 0)] = 1

    if np.amax(color_binary) < 200:
        color_binary = (color_binary / np.amax(color_binary)) * 255

    return color_binary

def calc_curvature(raw_x, raw_y):
    # Shoehorn into Udacity framework
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension

    fit = np.polyfit(raw_y*ym_per_pix, raw_x*xm_per_pix, 2)
    y_eval = np.max(raw_y) # Bottom of image. Y-value closest to the car.
    curverad = ((1 + (2*fit[0]*y_eval*ym_per_pix + fit[1])**2)**1.5) / np.absolute(2*fit[0])
    return curverad


def zero_left_right(img_bin_warp):
    np.where(img_bin_warp <= 100, img_bin_warp, 0)
    np.where(img_bin_warp > 100, img_bin_warp, 255)
    img_shape = img_bin_warp.shape
    zero_right = img_bin_warp.copy()
    zero_left = img_bin_warp.copy()
    zero_right[0:img_shape[0], img_shape[1]/2:img_shape[1]] = 0
    zero_left[0:img_shape[0], 0:img_shape[1]/2] = 0

    zero_right[0:img_shape[0], 450:img_shape[1]] = 0
    zero_left[0:img_shape[0], 0:900] = 0


    return zero_right, zero_left


def calc_fit(zero_left, zero_right):

    img_shape = zero_left.shape


    # Get fit parameters for left lane
    left_raw_all = np.nonzero(zero_right)
    left_raw_y = left_raw_all[0]
    left_raw_x = left_raw_all[1]
    #left_fit_fofx = np.polyfit(left_raw_x, left_raw_y, 2) # f(x), not f(y)
    left_fit_fofy = np.polyfit(left_raw_y, left_raw_x, 2) # f(y), not f(x)

    # Get fit parameters for right lane
    right_raw_all = np.nonzero(zero_left)
    right_raw_y = right_raw_all[0]
    right_raw_x = right_raw_all[1]
    #right_fit_fofx = np.polyfit(right_raw_x, right_raw_y, 2) # f(x), not f(y)
    right_fit_fofy = np.polyfit(right_raw_y, right_raw_x, 2) # f(y), not f(x)

    # Calculate curvature of lanes in meters
    right_curverad = calc_curvature(right_raw_x, right_raw_y)
    left_curverad = calc_curvature(left_raw_x, left_raw_y)
    left_curverad = int(left_curverad)
    right_curverad = int(right_curverad)
    avg_curverad = int((left_curverad + right_curverad) / 2)

    # Make fit to draw lines on image
    y_vals = np.linspace(0, img_shape[0]-1)
    x_vals = np.linspace(0, img_shape[1]-1)
    y_vals = y_vals.astype(int)
    x_vals = x_vals.astype(int)
    left_fit_xvals = left_fit_fofy[0]*y_vals**2 + left_fit_fofy[1]*y_vals + left_fit_fofy[2]
    right_fit_xvals = right_fit_fofy[0]*y_vals**2 + right_fit_fofy[1]*y_vals + right_fit_fofy[2]
    left_fit_xvals = left_fit_xvals.astype(int)
    right_fit_xvals = right_fit_xvals.astype(int)


    return left_fit_xvals, right_fit_xvals, y_vals, avg_curverad, left_curverad, right_curverad


def calc_offset(left_fit_xvals, right_fit_xvals, img_size):
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    lane_pos_left = left_fit_xvals[-1] * xm_per_pix
    lane_pos_right = right_fit_xvals[-1] * xm_per_pix
    lane_center = (lane_pos_right + lane_pos_left) / 2
    image_center = (img_size[0]/2) * xm_per_pix
    offset = image_center - lane_center
    return offset

def shade_lane(img_bin_warp, left_fit_xvals, right_fit_xvals, y_vals):
    global polygon_points_old
    print(polygon_points_old)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img_bin_warp).astype(np.uint8)
    img_color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fit_xvals, y_vals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_xvals, y_vals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(img_color_warp, np.int_([pts]), (0,255, 0))

    return(img_color_warp)

def annotate_image(newwarp, img, avg_curverad, offset):
    img = img.astype(np.uint8)
    newwarp = newwarp.astype(np.uint8)
    img_result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    # Put text on image
    textstr1 = 'Lane curvature = ' + str(avg_curverad) + ' m'
    textstr2 = 'Position = ' + str(offset) + ' m'
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_result,textstr1,(400,100), font, 1,(255,255,255),2)
    cv2.putText(img_result,textstr2,(400,125), font, 1,(255,255,255),2)
    return img_result

def make_src_dst(img_shape):
    # Define src points for transform
    offset_for_obscuration = 50
    tl_src = (585, 450) # top left
    tr_src = (720, 450) # top right
    br_src = (1150, img_shape[0] - offset_for_obscuration) # bottom right
    bl_src = (225,img_shape[0] - offset_for_obscuration) # bottom left
    src = np.float32([[tl_src, tr_src, br_src, bl_src]])

    tl_src = (560, 475) # top left
    tr_src = (735, 475) # top right
    br_src = (1150, img_shape[0] - offset_for_obscuration) # bottom right
    bl_src = (225,img_shape[0] - offset_for_obscuration) # bottom left
    src = np.float32([[tl_src, tr_src, br_src, bl_src]])


    # Define dst points for transform
    tl_dst = [320, 0] # top left
    tr_dst = [960, 0] # bottom left
    br_dst = [960, 720] # top right
    bl_dst = [320, 720] # bottom right OK
    dst = np.float32([[tl_dst, tr_dst, br_dst, bl_dst]])
    return src, dst

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
       # # was the line detected in the last iteration?
       # self.detected = False
       # # x values of the last n fits of the line
       # self.recent_xfitted = []
       # #average x values of the fitted line over the last n iterations
       # self.bestx = None
       # #polynomial coefficients averaged over the last n iterations
       # self.best_fit = None
       # #polynomial coefficients for the most recent fit
       # self.current_fit = [np.array([False])]
       # #radius of curvature of the line in some units
       # self.radius_of_curvature = None
       # #distance in meters of vehicle center from the line
       # self.line_base_pos = None
       # #difference in fit coefficients between last and new fits
       # self.diffs = np.array([0,0,0], dtype='float')
       # #x values for detected line pixels
       # self.allx = None
       # #y values for detected line pixels
       # self.ally = None

        self.left_fit_xvals = []
        self.right_fit_xvals = []
        self.y_vals = []
        self.left_curverad = []
        self.right_curverad = []
        self.numel_to_store = 11
        self.image_index = 0



def proc_pipeline(objpoints, imgpoints, img, verbose, video = '', outdir = '', name = ''):
    """ Process verbose image pipline on an image with intermediate states plotted
    and saved.

    Args:
        objpoints:
        imgpoints:
        images:

    Returns:

    1. Compute the camera calibration matrix and distortion coefficients given a set of
    chessboard images.

    2. Apply a distortion correction to raw images.

    Use color transforms, gradients, etc., to create a thresholded binary image.

    Apply a perspective transform to rectify binary image ("birds-eye view").

    Detect lane pixels and fit to find the lane boundary.

    Determine the curvature of the lane and vehicle position with respect to center.

    Warp the detected lane boundaries back onto the original image.

    Output visual display of the lane boundaries and numerical estimation of lane
    curvature and vehicle position.
    """

    # Calculate parameters for later use.
    img_shape = img.shape
    img_size = (img_shape[1],img_shape[0])
    src, dst = make_src_dst(img_shape)
    img_undistort = undistort_image(objpoints, imgpoints, img)
    M, Minv = get_warp_params(img, src, dst)


#    img_roi = region_of_interest(img_undistort, src)
#    img_bin = make_binary_image(img_roi)


    img_bin = make_binary_image(img_undistort)
    img_roi = region_of_interest(img_bin, src)



    img_bin_warp = warper(img_roi, M)
    zero_right, zero_left = zero_left_right(img_bin_warp)
    left_fit_xvals, right_fit_xvals, y_vals, avg_curverad, left_curverad, right_curverad = calc_fit(zero_left, zero_right)
    offset = calc_offset(left_fit_xvals, right_fit_xvals, img_size)
    img_color_warp = shade_lane(img_bin_warp, left_fit_xvals, right_fit_xvals, y_vals)
    newwarp = warper(img_color_warp, Minv)
    img_result = annotate_image(newwarp, img, avg_curverad, offset)


    if len(Lanes.left_curverad) < Lanes.numel_to_store:
        Lanes.left_fit_xvals.append(left_fit_xvals)
        Lanes.right_fit_xvals.append(right_fit_xvals)
        Lanes.y_vals.append(y_vals)
        Lanes.left_curverad.append(left_curverad)
        Lanes.right_curverad.append(right_curverad)

    if len(Lanes.left_curverad) >= Lanes.numel_to_store:
        print(' ')
        print('***Starting from fully-populated list')
        print(' ')

        # Update the list with new values
        image_index = Lanes.image_index
        Lanes.left_fit_xvals[image_index] = left_fit_xvals
        Lanes.right_fit_xvals[image_index] = right_fit_xvals
        Lanes.y_vals[image_index] = y_vals
        Lanes.left_curverad[image_index] = left_curverad
        Lanes.right_curverad[image_index] = right_curverad

        # Find index of median radius of curvature for both left and right
        median_left_curverad = statistics.median(Lanes.left_curverad)
        median_right_curverad = statistics.median(Lanes.right_curverad)
        left_index = Lanes.left_curverad.index(median_left_curverad)
        right_index = Lanes.right_curverad.index(median_right_curverad)
        print('Left, Right index = ', left_index, right_index)

        # Get fit values associated with the curvature
        left_fit_xvals = Lanes.left_fit_xvals[left_index]
        right_fit_xvals = Lanes.right_fit_xvals[right_index]
        print('Left, Right fit = ', left_fit_xvals, right_fit_xvals)


        # y_vals is always the same
        y_vals = Lanes.y_vals[left_index]

        # Calculate the remainder of the pipeline with new parameters
        offset = calc_offset(left_fit_xvals, right_fit_xvals, img_size)
        img_color_warp = shade_lane(img_bin_warp, left_fit_xvals, right_fit_xvals, y_vals)
        newwarp = warper(img_color_warp, Minv)
        img_result = annotate_image(newwarp, img, avg_curverad, offset)

        # Calculate new index position. Ensure we are wrapping when we get to end.
        Lanes.image_index = Lanes.image_index + 1
        if Lanes.image_index == Lanes.numel_to_store:
            Lanes.image_index = 0

        print('Offset = ', offset)
        print('Index = ', Lanes.image_index)

    if verbose:
        print('Entering verbose mode')

        img_roicrop = region_of_interest(img, src)
        img_newwarp = warper(img_roicrop, M)


        #img_roicrop_warp = warper(img_roi, M)
        #img_newwarp = warper(img_roicrop_warp, Minv)
        # Make figure of all intermediate results
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3, figsize=(30, 12))
        ax1.imshow(img_undistort)
        ax1.set_title('Undistorted original image')
        ax2.imshow(img_roicrop)
        ax2.set_title('ROI-cropped image')
        ax3.imshow(img_newwarp)
        ax3.set_title('Warped color image after cropping')
        ax4.imshow(img_bin, cmap = 'gray')
        ax4.set_title('Binary image')
        ax5.imshow(img_roi, cmap = 'gray')
        ax5.set_title('Binary image after ROI cropping')
        ax6.imshow(img_bin_warp, cmap = 'gray')
        ax6.set_title('Warp binary')

        ax7.imshow(zero_right, cmap = 'gray')
        ax7.set_title('Left lanes for fitting')
        ax8.imshow(zero_left, cmap = 'gray')
        ax8.set_title('Right lanes for fitting')
        ax9.set_title('No image')



        ax10.imshow(img_bin_warp, cmap = 'gray')
        ax10.plot(left_fit_xvals, y_vals, c='g', linewidth = 2)
        ax10.plot(right_fit_xvals, y_vals, c='b', linewidth = 2)
        ax10.set_title('Warp of binary and fit of lanes')
        ax11.imshow(img_color_warp)
        ax11.set_title('Lane shaded on binary image')
        ax12.imshow(img_result)
        ax12.set_title('Annotated, undistorted image')

        plt.show()
        out_path = os.path.join(outdir, name + '_total_results' + '.png')
        print(' ')
        print('outdir = ', outdir)
        print('name = ', name)
        print('out_path = ', out_path)
        print(' ')
        f.savefig(out_path, bbox_inches='tight', format='png')

        # Show large version of annotated final image
        cv2.imshow('Annotated final image', img_result)
        cv2.waitKey(100)
        cv2.destroyAllWindows()
        out_path = os.path.join(outdir, name + '_annotatedfinal' + '.jpg')
        cv2.imwrite(out_path, img_result)

    return img_result


def proc_video_pipeline(img):
    result = proc_pipeline(objpoints, imgpoints, img, False, video = 1, outdir = '', name = '')
    return result

def main():
    """
    Docstring
    """
    global objpoints
    global imgpoints
    global Lanes
    global img_count
    global polygon_points_old
    polygon_points_old = None
    Lanes = Line()
    img_count = 0

    dir_cal_images = os.path.join('.', 'camera_cal')
    dir_test_images = os.path.join('test_images')
    dir_output_images = os.path.join('output_images')

    proc_distortion_data = 0
    proc_pipeline_cal_images = 0
    proc_pipeline_test_images = 0
    proc_pipeline_video = 1
    verbose = False

    if proc_distortion_data == 1:
        objpoints, imgpoints = compute_camera_cal(dir_cal_images)
        pickle_data = {}
        pickle_data["objpoints"] = objpoints
        pickle_data["imgpoints"] = imgpoints
        pickle.dump( pickle_data, open("calibration_data.p", "wb" ))

    pickle_data = pickle.load(open("calibration_data.p","rb"))
    objpoints = pickle_data["objpoints"]
    imgpoints = pickle_data["imgpoints"]

    if proc_pipeline_cal_images == 1:
        search_phrase = os.path.join(dir_cal_images, '*.jpg')
        images = glob.glob(search_phrase)
        for fname in images:
            img = cv2.imread(fname)
            img_undistort = undistort_image(objpoints, imgpoints, img)
            curr_name = fname[-9:-4]
            out_path = os.path.join(dir_output_images, curr_name + '_undistorted' + '.jpg')
            cv2.imshow('Undistorted checkerboard', img_undistort)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
            cv2.imwrite(out_path, img_undistort)

    if proc_pipeline_test_images == 1:
        search_phrase = os.path.join(dir_test_images, '*.jpg')
        images = glob.glob(search_phrase)
        for fname in images:
            img = cv2.imread(fname)
            curr_name = fname[-9:-4]
            ret_img = proc_pipeline(objpoints, imgpoints, img, verbose, outdir = dir_output_images, name = curr_name)

    if proc_pipeline_video == 1:
        output = 'project_video_processed.mp4'
        in_clip = VideoFileClip("project_video.mp4")
        out_clip = in_clip.fl_image(proc_video_pipeline)
        out_clip.write_videofile(output, audio=False)




if __name__ == "__main__":
    main()
