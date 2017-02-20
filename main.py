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
    print(img.shape)
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
    kernel_size = 5
    img_blur_gray = gaussian_blur(gray_image, kernel_size)

    # Canny edge detection
    low_threshold = 50
    high_threshold = 150
    img_canny_blur_gray = canny(img_blur_gray, low_threshold, high_threshold)

    return img_canny_blur_gray

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image



def get_warp_params(img):
    """
    Docstring
    """
     # Define src matrix
    img_shape = img.shape
    img_size = (img_shape[1],img_shape[0])
    imshape = img.shape
    vert1 = (550, 450) # top left
    vert2 = (720, 450) # top right
    vert3 = (150,imshape[0]-50) # bottom left
    vert4 = (1150, imshape[0]-50) # bottom right
    src = np.float32([[vert1, vert2, vert3, vert4]])
    #print('binvert', vertices)
    #proc_img = region_of_interest(proc_img, vertices)
    #print('new vertices', src)
    #
    # Define src matrix vertices to mask and pull out lanes
    #vert1 = (.42 * img_shape[1], .67 * img_shape[0]) # bottom left
    #vert2 = (.58 * img_shape[1], .67 * img_shape[0]) # top left
    #vert3 = (0 * img_shape[1],img_shape[0]) # top right
    #vert4 = (1 * img_shape[1], img_shape[0]) # bottom right
    #src = np.float32([[vert1, vert2, vert3, vert4]])
    #print('copied src', src)



    # Define dst matrix for warping
    dst = np.float32([[0,0],[img_shape[1],0],[0,img_shape[0]],[img_shape[1],img_shape[0]]])

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

def proc_pipeline(objpoints, imgpoints, img, save_interm_results = 0, name = '',
    outdir = ''):
    """ Process an image pipline on an image

    Args:
        objpoints:
        imgpoints:
        images:

    Returns:

    Compute the camera calibration matrix and distortion coefficients given a set of
    chessboard images.

    Apply a distortion correction to raw images.

    Use color transforms, gradients, etc., to create a thresholded binary image.

    Apply a perspective transform to rectify binary image ("birds-eye view").

    Detect lane pixels and fit to find the lane boundary.

    Determine the curvature of the lane and vehicle position with respect to center.

    Warp the detected lane boundaries back onto the original image.

    Output visual display of the lane boundaries and numerical estimation of lane
    curvature and vehicle position.
    """
    #print(objpoints, imgpoints)
    #print(type(objpoints), type(imgpoints))

    cv2.imshow('img', img)
    cv2.waitKey(500)


    # Undistort image
    proc_img = undistort_image(objpoints, imgpoints, img)
    if save_interm_results:
        cv2.imshow('img', proc_img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        print(name)
        out_path = os.path.join(outdir, name + '_undistorted' + '.jpg')
        cv2.imwrite(out_path, proc_img)


#     Mask image
#     imshape = img.shape
#     vert1 = (550, 450) # top left
#     vert2 = (720, 450) # top right
#     vert3 = (1150, imshape[0]-50) # bottom right
#     vert4 = (150,imshape[0]-50) # bottom left
#     vertices = np.array([[vert1, vert2, vert3, vert4]], dtype=np.int32)
#     print('binvert', vertices)
#     proc_img = region_of_interest(proc_img, vertices)

    # Make binary image
    proc_img = make_binary_image(proc_img)
    if save_interm_results:
        cv2.imshow('img', proc_img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        out_path = os.path.join(outdir, name + '_bin' + '.jpg')
        cv2.imwrite(out_path, proc_img)

    # Make perspective transformed image
    M, Minv = get_warp_params(img)
    proc_img = warper(img, M)
    if save_interm_results:
        cv2.imshow('img', proc_img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        out_path = os.path.join(outdir, name + '_perspectivetransformed' + '.jpg')
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