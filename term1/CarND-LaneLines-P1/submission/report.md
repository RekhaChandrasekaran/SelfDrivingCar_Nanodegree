# **Finding Lane Lines on the Road**

---

**Finding Lane Lines on the Road**


---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of following steps.
1. I converted input image to grayscale.
2. I applied smoothening using GaussianBlur and kernel size of 5.
3. I used Canny Edge detection with low_threshold=50 and high_threshold=150, with the suggested ratio of 1:3. These values gave better results.
4. I used a hard-coded region mask to form a trapezoid shape starting from the center of the image and expanding to both sides at the bottom.
5. Then I used Hough transform with the following parameters to find line segments in the ROI region.
rho=5, theta=5 (grid size), threshold=50 (number of votes required), min_line_length=10, max_line_gap=10.
I experimented with several different values and found that this combination worked better for all the input images, especially for the images with curved lines.    
**Extrapolated Lane Lines**  
I defined a function draw_lines_extrapolated and extended draw_lines functionality.
1. I calculated slope for every line segment that was detected by Hough lines function.
2. I filtered lines with slope of angle 30 - 70 degree, with the assumption that the lane lines would appear within this range. This eliminated some spurious line segments.
3. With the knowledge that left lane line has positive slope and right lane line has negative slope, I split the line segments into two separate buckets accordingly.
4. I calculated average slope and bias for left and right lane line using the above buckets. i.e. left lane slope is the average slope of all detected lines with a positive slope.
5. Once I found the slope and bias for both left and right lanes, I used the y1, y2 values of trapezoid ROI, to calculate x1, x2 values for both line segments.  
This way I was able to extend the line to fill the region of interest.  
x = (y-bias)/slope
6. Then I simply drew both line segments on the input image.



### 2. Identify potential shortcomings with your current pipeline

The major shortcoming in this implementation are the hard-coded values. I think that the thresholds used for Canny Edge Detection will fail in different lighting conditions. Also the hard-coded values for the region of interest and the parameters to detect lines will also fail, if the input was of different shape or when the lane lines are curved.  
Also it takes a few seconds to process the video images, so I think this implementation is not suitable for real-time application.

### 3. Suggest possible improvements to your pipeline

* First, I want to make the code robust by handling exception scenarios.   
* Then I wish to define the thresholds and parameters that generalize to all possible conditions. i.e. remove hard-coded parameters.  
* Instead of grayscale, use HSV or other color space and detect bright white / yellow lane lines.
* Improve processing time by efficiently using numpy operations and avoiding for loops.  
* I want to use advanced techniques like Deep Learning for lane detection.
