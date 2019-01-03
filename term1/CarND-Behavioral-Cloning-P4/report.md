# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report  

Reference:  
[End to End Learning for Self Driving Cars : Nvidia](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)


[//]: # (Image References)

[image1]: ./report_images/Nvidia.jpg "Nvidia Model"
[image2]: ./report_images/model_arch.jpg "Model Architecture"
[image3]: ./report_images/steering.jpg "Steering angle"
[image4]: ./report_images/sample.jpg "Sample"

### Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used Nvidia's architecture (see Reference) as the base model for my project. It consists of several convolution layers and Fully connected layers. This architecture was very effective for this problem without much modification. I focused on tuning hyperparameters and dropout layers to prevent overfitting.
(model.py line 66)  

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.  

Architecture of Nvidia model.  

![alt text][image1]



#### 2. Attempts to reduce overfitting in the model

To reduce overfitting, I added BatchNormalization to the convolution layers instead of Dropout. Literature suggests that BN is effective to reduce overfitting in CNN layers. I also added Dropout layers and l2 regularization on the Fully connected layers. These hyperparameters were selected upon trial and error.  

The accuracy and loss metrics didn't provide a clear perspective of the model's performance on the test track. So I decided to use a small validation dataset of size 10%. This validation set it used to determine whether the model was over or underfitting. The final model showed similar loss values for both training and validation sets at each epoch and the loss kept decreasing.  
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Using ELU layer (as suggested in slack channel) proved to be effective in reducing the training time.

#### 4. Appropriate training data

I couldn't collect proper training data using the online simulator since it was lagging. And I was unable to run the simulator on my low end machine. I chose to use the data provided by Udacity to train my model.  
The data was collected by driving the car around the track 1 and capturing images from three separate cameras: left, center, and right. In total, 24,108 images were available for training.  

The main problem with this training data is that most of the images had steering angle close to 0. Without data augmentation, the model will be biased towards driving straight in all circumstances.  

![alt text][image3]

As suggested in the class, I added a steering offset of 0.25 to the left and right camera images and used them to train my model.
This is done so that we have data from car being on the left and right sides of the lane, and by adding an offset to the left and right steering angles we can train our model to correct back towards the center of the lane.  

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

As stated above, I used Nvidia's architecture as a base for my model.
I trained the model for 15 epochs, I found that the model performed better when it was trained for at least 10 epochs.  

The final step was to run the simulator to see how well the car was driving around track one. The driving is not smooth, it could be because of the lag in the online simulator. I repeated testing for a few times but the lag persisted.
The vehicle was able to drive autonomously around track one without leaving the road.

#### 2. Final Model Architecture

My final model architecture:

![alt text][image2]

As shown in the image, the model has a lambda layer for normalization, five convolutional layers with BatchNormalization and ELU layers. The kernel size of the CNN layers were chosen as per the reference paper. The final layers of the model consists of a Flatten layer and few Dense layers. Dropout was added in these final layers to prevent overfitting. The last layer of the model is a Dense layer with one node as the model outputs the steering angle which is a continuous value.
This architecture has 2,883,091 trainable parameters. The training was fast on the GPUs provided in the workspace. The model was trained for 15 epochs with batch size 32 (image generator will add another 32 augmented images) and 'adam' optimizer to minimize mean squared error 'mse'.  

#### 3. Creation of the Training Set & Training Process

The major task of this project is to augment data.  
* To utilize left and right camera images, I added a steering offset of 0.25 as suggested in the class. (+steering_offset for left image, -steering_offset for right image).  
* Then I flipped all the available training images to provide more training samples. The steering angle is also flipped to match the image. After augmenting, the batch is shuffled so that no bias was introduced because of ordering of the images.
* Then cropped the sky in the top portion of the image and also the car's hood at the bottom.  
* I resized the image to 160x320 and modified the input layer for the Nvidia model architecture as well.

Example of the processed image.  
![alt text][image4]

As stated above, the original dataset has a bias towards steering angle 0. To balance the selection of images from all three cameras, I applied the following step. At every batch, I limited the number of images labelled steering angle close to 0, to at most half of the batch size. Exceeding that limit, a random image with higher steering angle will be chosen from the training data and included in the batch. This was very effective in training model to drive around sharp edges.

#### 4. Results
The test run of the model driving autonomously on track one is shared here.  

[Video](https://youtu.be/CGeSkvc2t3A)  
[run1](https://youtu.be/Y-ubXUPZ7wU)

#### 5. Future work
I would like the car to drive smooth and on the center of the lane for most part. I think by collecting more training data focused on center lane driving and recovery from the sides to the center will make the model better. I also want to try adaptive throttle as that's natural driving behavior. When the steering angle is steep, we will naturally reduce the speed. The steering angle we apply will be based on the acceleration as well. By adjusting throttle based on steering angle will be a good starting point. Also, there is a lag in the online simulator and I think testing the model on local GPU will give better results.  
At last, the model drove well on track one but not on track two. The model has not seen the any images on track two. I want to collect track two data and train it as well.
