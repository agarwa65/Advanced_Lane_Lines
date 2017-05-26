## README 
---
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/test_undist.png "Road Transformed"
[image3]: ./examples/srb_image.png "Binary Example"
[image4]: ./examples/warped_image.png "Warp Example"
[image5]: ./examples/color_fit_lines.png "Fit Visual"
[image6]: ./examples/example_output.png "Output"
[video1]: ./project_video_output.mp4 "Video"

---

### README

### Camera Calibration

The code for this step is contained in the file called `cameraCaliberation.py` under utils folder. The output of the function is shown in cell 2 and 3 of the lane_detection_pipeline.ipynb file.   

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Image Undistortion

Using mtx and dist found in previous step of camera caliberation. I applied the distortion correction to one of the test images like the one below using cv2.undistort function.

![alt text][image2]

#### 2. Image Thresholding

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in `threshold.py`).  Here's an example of my output for this step.  

![alt text][image3]

#### 3. Perspective Transform

The code for my perspective transform includes a function called `img_warped()`, which appears in the file `cameraCaliberation.py` (utils/cameraCaliberation.py) (and used in the 5th code cell of the IPython notebook).  The `img_warped()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 592, 451      | 250, 0        | 
| 254, 688      | 250, 720      |
| 1056, 688     | 1000, 720      |
| 684, 451      | 1000, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane Pixels are identified by adding up pixel values along each column of the image. Then using window based search method the peaks in the histogram are identifies which are good indicators of the x-position of the base of the lane lines. This is then used as a starting point to search for lines with a margin of 100 px. This code is contained in function pipeline() of the notebook.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To find radius of curvature, the lane polynoimial calculated from the pipeline() function for both lanes in world coordinates was used.

The conversion from pixel to world space is as follows:
ym_per_pix = 4.0/72.0 # meters per pixel in y dimension
xm_per_pix = 3.7/688.0 # meters per pixel in x dimension

The curvature formula can be found in pipeline() function. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in pipeline() function in my code in `lane_detection_pipeline.ipnyb`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The code does not perform very well on challenge video as the pipeline is not robust enough for varied illumination color shades in challenge videos. The pipeline is based on color and gradient thresholding.

Tuning the parameters manually for cv2 functions was very difficult which enforces the importance of machine learning and deep learning tools. 
