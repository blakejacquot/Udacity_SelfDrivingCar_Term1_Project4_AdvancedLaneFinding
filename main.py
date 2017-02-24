"""
    Docstring
"""

# Import standard libraries
import glob
import os
import math
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import math
import pickle
#from scipy.signal import find_peaks_cwt

#Import non-standard libraries
import cv2

import time


def compute_camera_cal(dir_cal_images):
    """Compute distortion parameters for chessboard jpg's in a directory

    Args:
        dir_cal_images: Directory of calibration chessboard images

    Returns:
        objpoints: List of 3d points in real world space.
        imgpoints: List of 2d points in image plane.

    """
    print('Computing camera calibration matrices')
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
        print('Three-channel image')
    else:
        ignore_mask_color = 255
        print('One-channel image')

    print(ignore_mask_color)
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

    # Display info
    print('src', src)
    print('dst', dst)

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
    img_warped = cv2.warpPerspective(img, M, img_size)
    return img_warped

def unwarp(img, Minv):
    img_shape = img.shape
    img_size = (img_shape[1],img_shape[0])
    return cv2.warpPerspective(img,Minv,img_size)
    return img

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

def mark_lane_pixels():
    """
    Docstring
    """
    pass

def calc_lane_curvature():
    """
    Docstring
    """
    pass

def warp_lanes_to_origimage():
    """
    Docstring
    """
    pass

def write_annotated_image():
    """
    Docstring
    """
    pass


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
#    gradx = abs_sobel_thresh(img_blur_gray, orient='x', sobel_kernel=ksize, thresh=(10, 255))
#    grady = abs_sobel_thresh(img_blur_gray, orient='y', sobel_kernel=ksize, thresh=(60, 255))
#    mag_binary = mag_thresh(img_blur_gray, sobel_kernel=ksize, mag_thresh=(40, 255))
#    dir_binary = dir_threshold(img_blur_gray, sobel_kernel=ksize, thresh=(.65, 1.05))

    gradx = abs_sobel_thresh(img_blur_gray, orient='x', sobel_kernel=ksize, thresh=(200, 255))
    grady = abs_sobel_thresh(img_blur_gray, orient='y', sobel_kernel=ksize, thresh=(200, 255))
    mag_binary = mag_thresh(img_blur_gray, sobel_kernel=ksize, mag_thresh=(200, 255))
    dir_binary = dir_threshold(img_blur_gray, sobel_kernel=ksize, thresh=(.95, 1.05))


    # Combine all the thresholding information
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1


    # Get hls channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h = hls[:,:,0]
    l = hls[:,:,1]
    s = hls[:,:,2]

    # Filter on s channel in coordination with the combined image
    s_binary = np.zeros_like(combined)
    s_binary[(s > 160) & (s < 255)] = 1
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.zeros_like(combined)
    color_binary[(s_binary > 0) | (combined > 0)] = 1

    return color_binary



def proc_pipeline(objpoints, imgpoints, img, save_interm_results = 0, name = '',
    outdir = ''):
    """ Process an image pipline on an image

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

    # Display original image
    cv2.imshow('img', img)
    cv2.waitKey(500)

    # Undistort image and save results.
    proc_img = undistort_image(objpoints, imgpoints, img)
    if save_interm_results:
        cv2.imshow('img', proc_img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        out_path = os.path.join(outdir, name + '_undistorted' + '.jpg')
        cv2.imwrite(out_path, proc_img)

    # Calculate parameters for later use.
    img_shape = img.shape
    img_size = (img_shape[1],img_shape[0])

    # Define src points for transform
    offset_for_obscuration = 50
    tl_src = (585, 450) # top left
    tr_src = (720, 450) # top right
    br_src = (1150, img_shape[0] - offset_for_obscuration) # bottom right
    bl_src = (225,img_shape[0] - offset_for_obscuration) # bottom left
    src = np.float32([[tl_src, tr_src, br_src, bl_src]])

    # Define dst points for transform
    tl_dst = [320, 0] # top left
    tr_dst = [960, 0] # bottom left
    br_dst = [960, 720] # top right
    bl_dst = [320, 720] # bottom right OK
    dst = np.float32([[tl_dst, tr_dst, br_dst, bl_dst]])


    roi_img = region_of_interest(proc_img, src)
    if save_interm_results:
        cv2.imshow('roi_img', roi_img)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()
        out_path = os.path.join(outdir, name + '_ROIimg' + '.jpg')
        cv2.imwrite(out_path, proc_img)

    proc_img = roi_img

    # Draw region of interest on image
    lines = np.zeros((4,1,4))
    lines[0,0,:] = np.array([tl_src[0], tl_src[1], tr_src[0], tr_src[1]]) #tl, tr
    lines[1,0,:] = np.array([br_src[0], br_src[1], tr_src[0], tr_src[1]]) #tr, br
    lines[2,0,:] = np.array([br_src[0], br_src[1], bl_src[0], bl_src[1]]) #br, bl
    lines[3,0,:] = np.array([tl_src[0], tl_src[1], bl_src[0], bl_src[1]]) #bl, tl

    # Draw the region of interest on undistorted image and save results.
    proc_img_temp = proc_img.copy()
    proc_img_temp = draw_lines(proc_img_temp, lines)
    if save_interm_results:
        cv2.imshow('img', proc_img_temp)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        out_path = os.path.join(outdir, name + '_drawROI' + '.jpg')
        cv2.imwrite(out_path, proc_img_temp)

    # Make perspective transformed image from region of interest image and save results
    M, Minv = get_warp_params(img, src, dst)
    proc_img_temp = warper(proc_img_temp, M)
    if save_interm_results:
        cv2.imshow('img', proc_img_temp)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        out_path = os.path.join(outdir, name + '_perspectivetransformed_drawROI' + '.jpg')
        cv2.imwrite(out_path, proc_img_temp)

    # Make binary image, crop to ROI, and save results.
    proc_img = make_binary_image(proc_img)
    if save_interm_results:
        cv2.imshow('img', proc_img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        out_path = os.path.join(outdir, name + '_bin' + '.jpg')
        cv2.imwrite(out_path, proc_img)



    #proc_img = region_of_interest(proc_img, src)
    #tl = (585, 450) # top left
    #tr = (720, 450) # top right
    #br = (1150, img_shape[0]) # bottom right
    #bl = (225,img_shape[0]) # bottom left
    #x,y
    #tl = (0, 0) # top left
    #tr = (img_shape[1], 0) # top right
    #br = (0, img_shape[0]) # bottom right
    #bl = (img_shape[1],img_shape[0]) # bottom left
    #vertices = np.float32([[tl, tr, br, bl]])
    #roi_image = region_of_interest(proc_img, vertices)
    #if save_interm_results:
    #    cv2.imshow('_binafterROI', proc_img)
    #    cv2.waitKey(500)
    #    cv2.destroyAllWindows()
    #    out_path = os.path.join(outdir, name + '_binafterROI' + '.jpg')
    #    cv2.imwrite(out_path, proc_img)





##     Get Hough lines, draw lines on image, trim to ROI, and save results.
#     lines = hough_lines(proc_img) # returns numpy.ndarray of shape (x, 1, 4)
#     proc_img = draw_lines(proc_img, lines)
#     vertices = np.array([[tl_src, tr_src, br_src, bl_src]], dtype=np.int32)
#     proc_img = region_of_interest(proc_img, vertices)
#     if save_interm_results:
#         cv2.imshow('img', proc_img)
#         cv2.waitKey(500)
#         cv2.destroyAllWindows()
#         out_path = os.path.join(outdir, name + '_drawHoughlines' + '.jpg')
#         cv2.imwrite(out_path, proc_img)

    # Make perspective transformed image on the Hough image.
    M, Minv = get_warp_params(img, src, dst)
    proc_img = warper(proc_img, M)
    if save_interm_results:
        cv2.imshow('img', proc_img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        out_path = os.path.join(outdir, name + '_drawHoughlines_transform' + '.jpg')
        cv2.imwrite(out_path, proc_img)

#    masked_canny_blur_gray = region_of_interest(canny_blur_gray, vertices)



    mark_lane_pixels()



    calc_lane_curvature()



    warp_lanes_to_origimage()


    write_annotated_image()


    return proc_img


def main():
    """
    Docstring
    """
    dir_cal_images = os.path.join('.', 'camera_cal')
    dir_test_images = os.path.join('test_images')
    dir_output_images = os.path.join('output_images')

    proc_distortion_data = 0
    proc_pipeline_cal_images = 0
    proc_pipeline_test_images = 1
    proc_pipeline_target_images = 0

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
            print('Processing ', fname)
            img = cv2.imread(fname)
            ret_img = proc_pipeline(objpoints, imgpoints, img, outdir = dir_output_images)
            curr_name = fname[-10:-4]
            out_pat, h = os.path.join(dir_output_images, curr_name + '.jpg')
            cv2.imwrite(out_path, ret_img)

    if proc_pipeline_test_images == 1:
        search_phrase = os.path.join(dir_test_images, '*.jpg')
        images = glob.glob(search_phrase)
        for fname in images:
            print('Processing ', fname)
            img = cv2.imread(fname)
            curr_name = fname[-9:-4]
            ret_img = proc_pipeline(objpoints, imgpoints, img, save_interm_results = 1,
                name = curr_name)

    if proc_pipeline_target_images == 0:
        print(dir_output_images)


if __name__ == "__main__":
    main()
