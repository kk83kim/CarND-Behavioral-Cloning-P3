# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


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

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
