# **Traffic Sign Recognition**

## Report

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/random_img_class.png "Random Images"
[image2]: ./report_images/class_distribution.png "Class Distribution"
[image3]: ./report_images/normalized.png "Normalized"
[image4]: ./report_images/data_augmentation.png "Data Augmentation"
[image5]: ./report_images/Roadwork.jpg "Road Work"
[image6]: ./report_images/No_Entry.jpg "No Entry"
[image7]: ./report_images/Go_Straight_or_Right.jpg "Go Straight or Right"
[image8]: ./report_images/Stop.jpg "Stop"
[image9]: ./report_images/Speed_Limit_70.jpg "Speed Limit 70"
[image10]: ./report_images/Pedastrian.jpg "Pedastrian"
[image11]: ./report_images/Priority_Road.jpg "Priority Road"
[image12]: ./report_images/Slippery_Road.jpg "Slippery_Road"

[image13]: ./report_images/prediction_1.png "Prediction 1"
[image14]: ./report_images/prediction_2.png "Prediction 2"
[image15]: ./report_images/prediction_3.png "Prediction 3"
[image16]: ./report_images/prediction_4.png "Prediction 4"
[image17]: ./report_images/prediction_5.png "Prediction 5"
[image18]: ./report_images/prediction_6.png "Prediction 6"
[image19]: ./report_images/prediction_7.png "Prediction 7"
[image20]: ./report_images/prediction_8.png "Prediction 8"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Report

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/RekhaChandrasekaran/SelfDrivingCar_Nanodegree/tree/master/term1/CarND-Traffic-Sign-Classifier-Project-P3)

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the **numpy** library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.    

**Random Image from each class**

![Random Image][image1]

**Distribution of classes in Training and Test Dataset**  

![Class Distribution][image2]

---

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

1. I decided to use the original color images instead of grayscale or any other individual channels. I think the color information in the traffic signs will be helpful for prediction. Generally red color indicates warning (e.g. stop), blue indicates information (e.g. Go ahead).  

2. I used normalization to rescale the pixel values between 0.0 and 1.0.  
Normalization is the general preprocessing technique used for feature scaling in Computer Vision. It accelerates convergence of the model and improves accuracy.
Future work, try other preprocessing techniques.  

![Normalized][image3]

3. I decided to generate additional data because the classes are imbalanced in both training and test dataset.
Then I used *ImageDataGenerator* from *Keras* preprocessing library to generate random fake data for data augmentation. I selected few transformations like rotation, shift and zoom. These transformations are randomly applied to the training images to create some variety in the data. I limited these transformations to a narrow degree, so that the resulting images doesn't deviate much from the original images.

Here is an example of an original image and an augmented image:

![Data Augmentation][image4]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution Layer 1     	| Kernel: 5x5 <br> stride: 1 <br> padding: 'same' <br> outputs: 32x32x32 	|
| RELU	1			|												|
| Max pooling	 1   	| Kernel: 3x3 <br> stride: 2 <br> dropout: 0.9 (keep_prob) <br>  outputs 16x16x32 				|
| Convolution Layer 2	    | Kernel: 5x5 <br> stride: 1 <br> padding: 'same' <br> outputs: 16x16x64      									|
| RELU	2			|												|
| Max pooling	 2   	| Kernel: 3x3 <br> stride: 2 <br> dropout: 0.9 (keep_prob) <br> outputs 8x8x64 				|
| Fully Connected	Layer 3	| dropout: 0.5 (keep_prob) <br> outputs 1024         									|
| RELU	3			|												|
| Fully Connected	Layer 4	| outputs 43         									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* As shown in class lectures, I used Softmax Cross Entropy for loss function and Adam Optimizer. Adam optimizer works better than gradient descent and it accelerates model convergence.
* I chose the 0.001 as learning rate, 128 for batch size. 128 was optimal since its not too small to affect training time or too large for the system to process.
* I trained the final model for 10 epochs. The trained model was overfitting and performed relatively poor in the validation data.  
* I also applied Regularization with beta 1e-6 to reduce overfitting.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The structure of CNN model trained was pretty standard in Computer Vision Deep learning models. I experimented with different kernel sizes for convolution layers, depth, number of nodes in FC layers and dropout parameters. I choose the above hyperparameters to achieve an validation accuracy of 93%.  

My final model results were:
* training set accuracy of **0.99**
* validation set accuracy of **0.93**
* test set accuracy of **0.93**

*Further Improvements*
* Early stopping and stopping the training after validation accuracy of 0.93.
* Use of pre-trained models.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![Web Images][image5] ![Web Images][image6]
![Web Images][image7] ![Web Images][image8]
![Web Images][image9] ![Web Images][image10]
![Web Images][image11] ![Web Images][image12]

Among these images, following signs will be easier to classify.
* Stop
* No Entry
* Priority Road
* Speed Limit 70

I think the following signs will be difficult to classify, because when I look at the compressed (32x32) images, they all look very similar and smudged.
* Roadwork
* Pedestrian
* Slippery Road

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Road work      		| Road work    									|
| No entry     			| No entry 										|
| Go straight or right			| Go straight or right										|
| Stop	      		| Stop				 				|
| Speed Limit 70			| Speed Limit 70     							|
| Pedestrian			| General Caution    							|
| Priority road			|  Priority road											|
| Slippery road			|  Road work											|


The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 75%. This comparably low to the accuracy on the test set of 93%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Softmax probabilities of top-5 predicted classes for each test image is given below.

![Web Images][image13] ![Web Images][image14]
![Web Images][image15] ![Web Images][image16]
![Web Images][image17] ![Web Images][image18]
![Web Images][image19] ![Web Images][image20]

When we look at the predicted probabilities, the model is very sure of its predictions. But it makes 2 mistakes here, and its something I suspected earlier.
* When we look at the 'No Entry' images, the top-3 predicted classes are 'No entry', 'No passing', 'Stop'. It makes sense as these signs have a red background and some white pixels in the middle.
* Similarly, for 'Go straight or right', the model predicts 'Turn left ahead', 'Keep right', 'Turn right ahead' along with the actual class. All these signs have white arrows indicating the direction.
* For 'Pedestrian' image, the model predicted 'General Caution' and 'Pedestrian' in the top-3 list. And for 'Slippery Road', it predicted 'Road work'.  
When we look at the compressed version of these images, they are all similar with a red triangle, and some black pixels smudged on a white background.  
Given that the validation accuracy is 93%, I think using a better compression method for these web images might improve the predictions.
