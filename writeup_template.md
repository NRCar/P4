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

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation. 

My project includes the following files:
* **project.ipynb** containing the code for the project
* **project.html** the saved html version of the jupyter notebok
* **output_images** containing the output image files [ more examples in the saved notebook] 
* **writeup.md** teh writeup
* **proccessed_project_video.mp4** the processed project video

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the IPython notebook located in "project.ipynb" in cell number: 2 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

<figure>
    <img src="https://github.com/NRCar/P4/blob/master/output_images/undistorted.png"/>
    <figcaption text-align: center>Undistorted Calibration Image</figcaption>
</figure>

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

<figure>
    <img src="https://github.com/NRCar/P4/blob/master/output_images/undistorted_test.png"/>
    <figcaption text-align: center>Undistorted Calibration Image</figcaption>
</figure>

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warpPerspective()`, which appears in Cell number 6 of the IPython notebook  The `warpPerspective()` function takes as inputs an image (`img`), and the harddcoded points to warp the image to birds eye view based or from the birdsye to unwarped based on the bool flag to_birds_eye

I verified that my perspective transform was working as expected by warping the test image as shown

<figure>
    <img src="https://github.com/NRCar/P4/blob/master/output_images/birds_eye.png" />
    <figcaption text-align: center>Warped birds eye view</figcaption>
</figure>

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image in the Cell number : 7 of the notebook.
I pull the binary threasholded S, L channels as well as a thresholded R&G channel for yellow.

I also applied the x sobel and direction sobel to generate the respective binary images.

```python
sobel = Get_Sobel_IMG(birds_eye)
    
    s_img = Get_S_Img(birds_eye)
    white = Get_L_Img(birds_eye)
    yellow = Get_Yellow(birds_eye)    
    
    lanes = np.zeros_like(yellow)
    lanes[((yellow == 1) | (white == 1)) & ((s_img == 1) | (sobel == 1))] = 1    
```

<figure>
    <img src="https://github.com/NRCar/P4/blob/master/output_images/binary_lanes.png" />
    <figcaption text-align: center>Bianry thresholded lanes</figcaption>
</figure>


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
I used the histogram function to identify the peaks in the left and right half of the images, to mark the lane start at the bottom of the image and then used the sliding window approach to identify the lane points and fit them into a polynomial as seen in cells 9 and 10 of the notebook


<figure>
    <img src="https://github.com/NRCar/P4/blob/master/output_images/histogram.png"  />
    <figcaption text-align: center>Histogram showing peaks</figcaption>
</figure>

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In cell 10 of the notebook i added functions center and radius to compute the curvature of the lane and the position of the car in meters
```python
def radius(leftx, rightx, ploty):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    return "Radius of curvature : %.2f m" % ((left_curverad+right_curverad)/2)

def center(leftx, rightx, image):
    xm_per_pix = 3.7/700
    
    midx = image.shape[1]/2
    lanex = (leftx[-1] + rightx[-1]) /2
    center_dist = (midx - lanex) * xm_per_pix
    dir = "left"
    if center_dist < 0 :
        dir = "right"
        center_dist = - center_dist   
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

At the end of the cell 10 we have teh function displayLanes which integrates the above pipeline and gives us the lanes on the image.

<figure>
    <img src="https://github.com/NRCar/P4/blob/master/output_images/lanes.png" />
    <figcaption text-align: center>Histogram showing peaks</figcaption>
</figure>

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./proccessed_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
My pipeline might fail on roads with lots of shadows or in tunnels or with traffic that crosses lanes. We could try other binay thresholding methods or magnitude sobel to identify the lanes better. 
