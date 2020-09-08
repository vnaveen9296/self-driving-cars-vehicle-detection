## Writeup Template
#### This is my report for the Project "Vehicle Detection and Tracking" from Udacity Self Driving Car Nanodegree program.


---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier such as Linear SVM classifier.
* Explore different color spaces as well as using other features such as binned color features, histograms of colors along with HOG feature vector. 
* Perform normalization of features and randomize training and test split.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Construct the pipeline and run it on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar_images.jpg
[image2]: ./output_images/hog_1.jpg
[image3]: ./output_images/hog_params_2.jpg
[image4]: ./output_images/sliding_windows.jpg
[image5]: ./output_images/test1_detections.jpg
[image6]: ./output_images/test6_detections.jpg
[image7]: ./output_images/heatmap.jpg
[image8]: ./examples/labels_map.png
[image9]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project that I used as a guide and a starting point.

You're reading it!

### Code Files
Here are the main files used for the project:

| File  | Description |
|:------:|:----------:|
| training_pipeline_threaded.py     | Contains training pipeline and inference pipeline and all the required methods such as feature extraction etc |
| training_pipeline.py      | Same as threaded version. Within threaded version I used multiple threads when looking for cars in sliding windows|
|video_pipeline.py | Main file for processing the video. It uses methods from training_pipeline_threaded.py file |
| All other files   | They are used for generating visualization or experimenting with different values |



### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in `training_pipeline_threaded.py` at line numbers 99-117 within a function named as `get_hog_features()`. This function is called during training as well as inference.  

For experimenting with various parameters of HOG and producing visualization I used the code in the file `get_hog.py`. Within the main method of this file, I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example of HOG visualization for randomly selected car and notcar images:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters such as `orientations`, `pixel_per_block` and `cells_per_block`. I have used the following values for these three parameters and produced the HOG visualizations to compare them for a randomly selected car and non car image


|		| orientations | pixel_per_block | cells_per_block |
|:------:|:-----------:|:---------------:|:---------------:|
|Values Tried | 9, 11	| 8, 16 | 2, 4 |

Here is the visualization of HOG for different parameters for a car and not car image. The first row below shows the car and not car images whereas 2nd and 3rd rows show the HOG visualization for these images respectively. The values used are specified as part of the title. For example (9,8,2) indicates `orientations=9`, `pixels_per_block=8` and `cells_per_block=2` and so on.
I can see that a value of 8 is better than 16 for `pixels_per_block`. However for other parameters, it is hard to find out which combination is better (among first 4 columns of HOG visualizations). And I chose (11,8,2) in my final pipeline.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `LinearSVC()` that is part of `sklearn.svm` module. The code for this is available in the method named `training_pipeline()` in the file `training_pipeline_threaded.py`. The steps involved in this are shown below
* Read the images (cars and noncars): I did this using a method called `get_data()` function (line # 363)  
* Extract features from car images and noncar images: I did this at line numbers 377 and 380
* Normalize data, prepare labels and randomly split the data into training and test sets: I used `StandardScaler()` as well as `train_test_split()` from `scikit-learn` package. The code for this step is available at line numbers 388-398
* Train the classifier: I did this at line numbers 403-406

At the end of the method `training_pipeline()`, I also saved the parameters used, the classifier and the scaler in a pickle file for later use.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search the bottom half of the image for window positions. I used multi scale windows of size (80,80), (96,96) and (128,128). I used different overlaps for different window sizes. An overlap of 0.5 for window size of (80,80) and 0.75 for window sizes (96,96), (128,128) are used. The reason for different overlaps is to keep the size of possible search windows to a reasonable value.  If the number of possible windows is large, it takes a long time for searching through these windows for cars. 

The following is the visualization of mutiscale windows I used. The code for creating this visualization is in the file `sliding_windows_viz.py`

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]
![alt text][image6]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap. I maintained a history of such heatmaps for several frames and integrated them at each frame and then thresholded that cumulative map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. The code for this is in `video_pipeline.py` at line numbers 43 to 56.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video (I generated these from frames of `test_video.py`  (frame numbers 11 to 15)):

### Here are four frames, their corresponding heatmaps, labels and with bounding boxes drawn on frames in the video series

![alt text][image7]




---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline constructed involved experimenting with various kinds of things such as color spaces, feature vector selection, parameters of different kinds of features etc. Also it is challenging to pick the search windows sizes, their overlap and selecting the different sizes for multiscale windows. The number of windows has a direct effect on the speed of processing. Choosing a large number of search windows resulted in very slow processing of frames (and video stream). All other steps such as the heatmaps, adding heat, integrating heatmaps over several frames and thresholding to avoid false positives also involved experimentation and are tuned for the current scenario. The vehicles farther in the images (or video frames) are well visible to eyes but not detected by this pipeline. The current implementation limits to searching for vehicles in the bottom portion of the images/frames. If the entire image is considered for searching vehicles, it may increase the false positive rate. The other problem is related to speed of the overall detection process. In order to ensure that the code can run in real time, choosing the right number of search windows without sacrifising detection accuracy is critical.