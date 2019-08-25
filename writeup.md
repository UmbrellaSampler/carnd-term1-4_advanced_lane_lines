## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./output_images/camera_calibration.jpg "Undistorted"
[image2]: ./output_images/original.jpg "Original"
[image21]: ./output_images/undistorted.jpg "Undistorted"
[image3]: ./output_images/perspective_transform.jpg "Warped"
[image4]: ./output_images/binary.jpg "Binary"
[image5]: ./output_images/histogram.jpg "Histogram"
[image6]: ./output_images/finding_lane_line_pixels.jpg "Finding lane lines"
[image7]: ./output_images/formula_radius.gif "Formula Radius"
[image8]: ./output_images/formula_poly.gif "Formula Poly"
[image9]: ./output_images/line_projection.jpg "Line Projection"
[video1]: ./finding_lane_lines.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in lines 26-46 of cell 1 in the jupyter notebook called `finding_lane_lines.ipynb`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I applied the distortion correction to one of the test images in line 1 of cell 4. The effect can be observed well at the edges of the undistorted image:
![alt text][image2]
![alt text][image21]



#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is included in cell 5 (`finding_lane_lines.ipynb`). First I defined the function `getTransformMatrix()` which returns the transformation matrix `M` and its inverse `Minv` as well as the source `src` and destination `dst` points. The transfomation is computed calling the `cv2.getPerspectiveTransform()` function. I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
        [[(width/2) - 63, height/2 + 100],
        [((width/6) - 20), height],
        [(width*5 / 6) + 60, height],
        [(width/2 + 65), height/2 + 100]])
    dst = np.float32(
        [[(width/4), 0],
        [(width/4), height],
        [(width*3 / 4), height],
        [(width*3 / 4), 0]])
```

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 577, 460      | 320, 0        | 
| 293, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 705, 460      | 960, 0        |

In line 19, the `perspectiveTransform()` function takes an image and the transformation matrix and applies the perspective transform by calling `cv2.warpPerspective()`.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart (function `overlaySrcAndDes()` in line 23) to verify that the lines appear parallel in the warped image.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds as well as a crop operation to generate a binary image. These operations are performed in the function `s_sx_threshold()` in line 1 of cell 6. First I convert to HLS image colors to seperate the s- and the l-channel. Then I apply a threshold on the the output of Sobel operation in x direction by calling `sobelXThreshold()` (defined in line 14) on the l-channel with a lower and upper threshold of `sx_thresh=(35,105)`. Right after a lower and upper threshold of `s_thresh=(140,200)` is applied on the s-channel. Finally, the combination of both threshold operations is cropped at the left and right edge of the image by 150 pixels (line 12).

The chosen threshold values turned out to perform best on the test images.

The result of the above descriped method produces a binary image as shown here:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff... kinda like this ;)

The procedure of finding the lane lines happens in two steps:

1. Finding the lane line pixels.
2. Fitting a second order polynomial.

The code defined to execute these steps can be found in a function called `find_lane_pixels_and_fit_poly()` in line 1 of cell 10. In lines 3-6 the lane line pixels are extracted. Line 9 shows the code for fitting these pixels resulting in `left_fit` and `right_fit` second order polyniomal fit coeficients. More details follow below.  

##### 4.1 Finding lane line pixels
The first step splits into two scenarios:

* Initialization using a "sliding window" technique.
* Decting pixels within a margin around the previous found lines.

The sliding window technique is chosen the find the lane line pixels from scratch without having any previous information about the line parameters. For this purpose a function `find_lane_pixels_sliding_window()` in line 1 of cell 8 is defined which takes a wraped binary camera image as an input. First an estimation about the left and right lane centers at the base of the image (refered to as `leftx_base` and `rightx_base` in the code) is made by creating histogram of the bottom half of the binary image like depicted in the following: 

![alt text][image5]

By finding the pixel values where the line has greatest density the left and the right lane center positions can be calculated. The lane centers serve now as a starting point for the "sliding window" technique. This technique is not descriped here as it is already shown in detail in the course. But below is a visualization that I captured during the implementation.

The second technique used for finding the lane line pixels uses previously found lines and searches within a certain margin to get the new line pixels.
The function `find_lane_pixels_around_last_fit` is located in line 4 in cell 9. A margin of 100 pixels was chosen.

![alt text][image6]

##### 4.2 Fitting a second order polymial

The curves are approximated by a sconde order polynomial fit on the left and right lane line pixels, respectively. The function `fit_poly()` in line 81 of cell 8 executes this operation.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius is calcualted is calculated in function `radius()` (cell 12 line 14) which takes the left and right lane fit parameters. The radius is calucalted using the formula 

![alt text][image7]

where A and B are the fit coefficients of the quadratic equation 

![alt text][image8]

The position with respect to the center is determined in line 1 of cell 12 by the function `car_position()`. It caluculates the difference between the car position which is `car_pos_x = width//2` and the lane center in pixels. In order to get the difference in meters I chose the regular lane width to be 3.7m (Wikipedia, for highway lanes in USA). If the function returns a negative (positive) result the car is located left (right) wrt. the lane center. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell 11 in my code with the function `project_lines_and_unwarp()`.  Here is an example of my result on a test image:

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./finding_lane_lines.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


The pipeline will likely fail in situations where the curvature gets to high and the lines bend to the left or right edge of the image. In this case my cropping approach described above with cut off the lines. Also the Sobel-x operation in x direction might poorly perform on bended lines where the end has high gradients in y direction. To improve the pipeline one could also take a Sobel-y operation into account and work with directed gradients.  

To overcome the problem with wobbly lines in the video one could test the result with respect to consistency of the left and write lane results. For example a check if both lines are parallel could be implemented. In case this check failes one could trigger a new line pixel detection using the "sliding window" technique. 
