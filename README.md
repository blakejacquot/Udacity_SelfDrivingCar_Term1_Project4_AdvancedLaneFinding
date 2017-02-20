##Udacity Self Driving Car
##Term 1, Project 4
---

**Advanced Lane Finding Project**
The goals / steps of this project are:
* Compute the camera calibration matrix and distortion coefficients given a set of
chessboard images. (DONE)
* Apply a distortion correction to raw images. (DONE)
* Use color transforms, gradients, etc., to create a thresholded binary image. (DONE)
* Apply a perspective transform to rectify binary image ("birds-eye view"). (Not Done)
* Detect lane pixels and fit to find the lane boundary. (Not Done)
* Determine the curvature of the lane and vehicle position with respect to center. (Not Done)
* Warp the detected lane boundaries back onto the original image. (Not Done)
* Output visual display of the lane boundaries and numerical estimation of lane curvature
and vehicle position. (Not Done)

[//]: # (Image References)
[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


[cal_1]: ./camera_cal/calibration1.jpg "Cal-1"
[test_2]: ./test_images/test2.jpg "Test-1"
[cal_1_undistort]: ./output_images/calibration1_undistorted.jpg "Undistorted_cal-1"
[test_2_undistorted]: ./output_images/test2_undistorted.jpg "Undistorted_test-1"
[test_2_bin]: ./output_images/test2_bin.jpg "bin_test-1"
[test_2_trans]: ./output_images/test2_perspectivetransformed.jpg "trans_test-1"





[cal_123]: ./camera_cal/calibration1.jpg "Cal-1"
![alt text][cal_123]


[t1p1_link]: https://github.com/blakejacquot/udacity_SDC_T1P1_LaneFinding/blob/master/P1.ipynb

##[Rubric](https://review.udacity.com/#!/rubrics/571/view) Responses


###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You are  reading it.


###Camera Calibration
####1. Briefly state how you computed the camera matrix and distortion coefficients.
The code for this step is in `compute_camera_cal` and `undistort_image` functions from
file `main.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the
chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y)
 plane at z=0, such that the object points are the same for each calibration image.
  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be
  appended with a copy of it every time I successfully detect all chessboard corners in
  a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of
  the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and
distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this
distortion correction to the test image using the `cv2.undistort()` function and
obtained this result:

Original image
![alt text][cal_1]

Undistorted image
![alt text][cal_1_undistort]

---
###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
Read in calibration parameters from pickle file and use `undistort_image` function from
file `main.py`. The undistortion is subtle, but present.

Original image
![alt text][test_2]

Undistorted image
![alt text][test_2_undistorted]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
My method borrows heavily from my [first project][t1p1_link] in this course.

The order of operations is:
1. Grayscale image.
2. Gaussian blur image with kernel of specified size.
3. Apply Canny edge detection.

Code for making binary image is in `make_binary_image` function of file `main.py`.

Here is an example of my output for this step:

Binary image
![alt text][test_2_bin]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Code for perspective transform includes a function called `warper()` from file `main.py`.
The function `warper()` takes an image parameters calculated from the `get_warp_params`
function. `get_warp_params` calculates the source and destination points for the transform.
Source points are selected to crop the original image to the area containing probably lane
lines.

An example of a transformed image is:

Transformed image
![alt text][test_2_trans]

TBD
TBD
TBD
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |
TBD
TBD
TBD


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I verified that my perspective transform was working as expected by drawing the `src` and
`dst` points onto a test image and its warped counterpart to verify that the lines appear
parallel in the warped image.

![alt text][image4]


Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like
this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the
function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

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

