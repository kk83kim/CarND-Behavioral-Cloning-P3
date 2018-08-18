# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_2016_12_01_13_30_48_287.jpg "Center Image"
[image2]: ./examples/left_2016_12_01_13_30_48_287.jpg "Left Image"
[image3]: ./examples/right_2016_12_01_13_30_48_287.jpg "Right Image"

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I used a convolution neural network publish by the Nvidia for end-to-end Deep Learning for Self-Driving Cars (https://devblogs.nvidia.com/deep-learning-self-driving-cars/).  In this model, the input layer is first normalized, followed by 5 convolution layers, followed by 3 fully connected layers.  Below table shows the details of each layers.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Normalized Input         		| 160x320x3 color image   							| 
| 1st Layer:  Convolution 5x5 filter | Depth = 24 |
| RELU					| |
|2nd Layer:  Convolution 5x5 filter | Depth = 36 |
| RELU					| |							|
|3rd Layer:  Convolution 5x5 filter | Depth = 48 |
| RELU					| |							|
|4th Layer:  Convolution 3x3 filter | Depth = 64 |
| RELU					| |							|
|5th Layer:  Convolution 3x3 filter | Depth = 64 |
| RELU					| |							|
| 6th Layer:  Fully connected		| outputs 100        									|
| 7th Layer:  Fully connected		| outputs 50        									|
| 8th Layer:  Fully connected		| outputs 1        									|

#### 2. Attempts to reduce overfitting in the model
The model was trained and validated on the provided dataset.  This dataset contained images from 3 cameras (center, left, and right).  For each camera, there were images from several laps driving clockwise and counter-clockwise.  The number of epochs were tuned to avoid training overfitting to the dataset.  The number of epochs used to train the model was 5.

#### 3. Model parameter tuning
The model used an adam optimizer, so the learning rate was not tuned manually

#### 4. Appropriate training data
I used the training data provided from the class.  This dataset contained images from 3 cameras (center, left, and right).  For each camera, there were images from several laps driving clockwise and counter-clockwise. This dataset was enough to train the model that could keep the vehicle on road very well.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
My first step was to get something working.  I first build a model with a single fully connected layer and tested to see everything was working.  Then, I improved the model by adding more convolutional layers, normalizing the data, and augmenting the data.  Everytime I made a change, I retrained the model and ran simulation to see if the change made an improvement.

#### 2. Final Model Architecture
The final model I used was the one published from Nvidia.  Discussion about this model is mentioned at the above seciont, 

#### 3. Creation of the Training Set & Training Process
The provided training images had images captured from 3 cameras:
![alt text][image2]
![alt text][image1]
![alt text][image3]

The image dataset contained driving in both (clockwise and counter-clockwise) directions for all three cameras.  In addition, it contained a lot vehicle recovering from the left / right sides of the road back to the center so that the vehicle would learn to stay center of the road.

Before training, I randomly shuffled the data set and put 20% of the data into a validation set.  I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the training / validation accuracy vs epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
