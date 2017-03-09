##Udacity Self Driving Car
##Term 1, Project 4
---

**Advanced Lane Finding Project**
The goals / steps of this project are:
* Compute the camera calibration matrix and distortion coefficients given a set of
chessboard images. (DONE)
* Apply a distortion correction to raw images. (DONE)
* Use color transforms, gradients, etc., to create a thresholded binary image. (DONE)
* Apply a perspective transform to rectify binary image ("birds-eye view"). (DONE)
* Detect lane pixels and fit to find the lane boundary. (DONE)
* Determine the curvature of the lane and vehicle position with respect to center. (DONE)
* Warp the detected lane boundaries back onto the original image. (DONE)
* Output visual display of the lane boundaries and numerical estimation of lane curvature
and vehicle position. (DONE)

[t1p1_link]: https://github.com/blakejacquot/udacity_SDC_T1P1_LaneFinding/blob/master/P1.ipynb

##[Rubric](https://review.udacity.com/#!/rubrics/571/view) Responses


###Writeup / README

All functions are contained within 'main.py'. The main pipeline is run with the 'proc_pipeline'
function. Only the obtaining of original distortion parameters is housed outside the pipeline.

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You are  reading it.

###Camera Calibration
####1. Briefly state how you computed the camera matrix and distortion coefficients.
The code for this step is in `compute_camera_cal` and `undistort_image` functions from
file `main.py`.

Make 'objpoints', which will hold the x, y, z chessboard coordinates. The variable 'imgpoints'
has the x, y pixel positions for each corner where there was a successful detection.

Use these two variables to compute camera calibration. Driving function is 'cv2.undistort'

Original image

<img src="./camera_cal/calibration1.jpg" width="70%">


Undistorted image

<img src="./output_images/tion1_undistorted.jpg" width="70%">

---
###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
Read in calibration parameters from pickle file and use `undistort_image` function from
file `main.py`. The undistortion is subtle.

Original image

<img src="./test_images/test2.jpg" width="70%">


Undistorted image

<img src="./output_images/test2_undistorted.jpg" width="70%">


####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
My method borrows heavily from my [first project][t1p1_link] in this course.

The general order of operations is:
1. Grayscale image.
2. Gaussian blur image with kernel of specified size.
3. Apply edge detection.

Code for making binary image is in `make_binary_image` function of file `main.py`.

Here is an example of my output for this step:

Binary image

<img src="./output_images/test2_bin.jpg" width="70%">


####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Code for perspective transform includes a function called `warper()` from file `main.py`.
The function `warper()` takes an image parameters calculated from the `get_warp_params`
function. `get_warp_params` calculates the source and destination points for the transform.
Source points are selected to crop the original image to the area containing lanes.

After choosing the source points for the transform, I sanity-checked results by looking at
the resulting region of interest on straight lane lines and the transform of this image.

Undistorted image with source points drawn:

<img src="./output_images/ines1_drawROI.jpg" width="70%">

Transformed image:

<img src="./output_images/ines1_perspectivetransformed_drawROI.jpg" width="70%">


I used the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 450 tl   | 320, 0        |
| 225, 670 bl   | 320, 720      |
| 1150, 670 br  | 960, 720      |
| 720, 450 tr   | 960, 0        |

tr = top right, bl = bottom left

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Make a copy of the binary, perspective-transformed image

Use 'zero_left_right' to get two binary, perspective-transformed images: one with left
half zeroed out and one with right half zeroed out. Use 'calc_fit' to get fits for both
lanes as well as curvature. Fitting is done with second-order polynomial in a f(y) rather
than f(x) manner. The reason is that the x-range is very compressed (same x-value maps to
lots of y values). But for every y-value there is one or few x-values.

Examples of fitting are found in composite images below.

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This falls out of 'calc_fit' function, which has a call to 'calc_curvature'. This defines
meters per pixel in x and y directions and then gets polyfit parameters and calculates
curvature.


####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is the last section 'proc_pipeline'. An example image is:

<img src="./output_images/test2_annotatedfinal.jpg" width="100%">

####6. Examples of pipeline operating on all provided test images.

The entire pipeline is in 'main.py' in 'proc_pipeline' function. Briefly it:

1. Calculates source and destination matrices for warping.
2. Undistorts image.
3. Crops to region of interest.
4. Makes warp parameters.
5. Makes this into binary image.
6. Warps binary image.
7. Fits lanes and calculates curvature.
8. Calculates offset.
9. Shade lane and dewarp.
10. Annotate final image.

Below are examples of running the entire pipeline on sample provided test images.

<img src="./output_images/ines1_total_results_xxx.png" width="100%"><br><br><br>
<img src="./output_images/ines2_total_results.png" width="100%"><br><br><br>
<img src="./output_images/test1_total_results_xxx.png" width="100%"><br><br><br>
<img src="./output_images/test2_total_results.png" width="100%"><br><br><br>
<img src="./output_images/test3_total_results.png" width="100%"><br><br><br>
<img src="./output_images/test4_total_results_xxx.png" width="100%"><br><br><br>
<img src="./output_images/test5_total_results_xxx.png" width="100%"><br><br><br>
<img src="./output_images/test6_total_results_xxx.png" width="100%"><br><br><br>

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why,
where the pipeline might fail and how I might improve it if I were going to pursue this
project further.

