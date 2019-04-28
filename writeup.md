# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center_driving.jpg "Center driving"
[image2]: ./images/recovery_left_dashed.jpg "Recovery from left"
[image3]: ./images/recovery_right.jpg "Recovery from right"
[image4]: ./images/recovery_right_dashed.jpg "Recovery from right"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* video.mp4 containing a recording of one lap driven autonomously

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The archtecture of my model is very close to the archtecture of the LeNet model. It has 2 convolutional layers with filter size (5,5) and 6 filters (model.py lines 71 and 74). After each convolutional layer a max pooling layer (filter size (2,2), stride 2) is applied (lines 72 and 75).
After the second max pooling layer follow 3 fully connected layers with 120, 84 and 1 neurons (lines 78, 80 and 82).

As activation functions for all layer 'relu' has been chosen. At the beginning of the model a lambda layer (line 69) normalizes the data and a cropping layer (line 70) crops the input images, so they don't contain the hood of the vehicle and the sky.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 73, 76, 79 and 81). The dropout rate was set to 0.5.

The model was trained and validated on different data sets to ensure that the model was not overfitting (line 96 and 101-102). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track (see video.mp4 for a recording of one lap).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 85).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to take a known architecture, that has already been proven to work well and that I had been working with in the previous projects.

My first step was to use a convolution neural network model similar to the LeNet model. The model delivers good results while still having a reasonable number of parameters. In the lecture model architecures had been presented (like AlexNet or GoogLeNet), which show a much better performance than LeNet, but are also much bigger and take longer to train.
The LeNet-like architecture, that I have chosen gets the job done and allowed me to train one epoch in about 15min. Thus I was able to make good progress.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I introduced dropout layers after each layer of the model with a dropout rate og 0.5 during training.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track in steep curves, but so far I had only used the images of the center camera for training. Using the images of the left and right camera as well, improved the results massively. The vehicle could now complete a complete lap, without falling of.

#### 2. Final Model Architecture

The final model architecture (model.py lines 69-82) consisted of a convolution neural network with the following layers and layer sizes:

* Convolution2D, num_filter=6, filter_shape=(5,5) ,activation='relu'
* MaxPooling2D, filter_shape=(2,2), stride=2
* Dropout, dropout_rate=0.5
* Convolution2D, num_filter=6, filter_shape=(5,5) ,activation='relu'
* MaxPooling2D, filter_shape=(2,2), stride=2
* Dropout, dropout_rate=0.5
* Flatten, output=(1,6006)
* Dense, output_size=120, activation='relu'
* Dropout, dropout_rate=0.5
* Dense, output_size=84, activation='relu'
* Dropout, dropout_rate=0.5
* Dense, output_size=1

This is the output of the `model.summary()` function:

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 61, 316, 6)        456
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 30, 158, 6)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 30, 158, 6)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 26, 154, 6)        906
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 13, 77, 6)         0
_________________________________________________________________
dropout_2 (Dropout)          (None, 13, 77, 6)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 6006)              0
_________________________________________________________________
dense_1 (Dense)              (None, 120)               720840
_________________________________________________________________
dropout_3 (Dropout)          (None, 120)               0
_________________________________________________________________
dense_2 (Dense)              (None, 84)                10164
_________________________________________________________________
dropout_4 (Dropout)          (None, 84)                0
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 85
=================================================================
Total params: 732,451
Trainable params: 732,451
Non-trainable params: 0
_________________________________________________________________

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. One lap was recorded clockwise and the second one counter-clockwise. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to react, when it had left the ideal driving trajectory and got too clsoe the the road border. I made sure to record these recovering events on straight track parts and in curves. I also recorded them in curves with red and white dashed and on the bridge, so that the model had seen recovery events in as many circumstances as possible. I alse made sure to recover the vehicles, when it got too close to the inside of a curve as well as on the outside of a curve.

These images show what a recovery looks like starting from the left and the right respectively:

![alt text][image3]
![alt text][image4]
![alt text][image5]

After the collection process, I had 27405 number of data points. I then preprocessed this data by normalizing the image values to a renage from -1 to 1. I also cropped them, to remove unnecessary parts of the image like the hood of the car and the sky.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. Using more would not improve the mean square error. I used an adam optimizer so that manually training the learning rate wasn't necessary.
