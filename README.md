# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:

- **model.py** : Containing the script to create and train the model
- **drive.py** : For driving the car in autonomous mode in the simulator (This is provided [Udacity](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/drive.py), I made a modificaion where I added a speed regulator.
- **model.h5** : Containing a trained convolution neural network.
- **writeup_report.md** : Summarizing the results
- **clone.py and clone2.py** : Test runs using LeNet and nVidia
- **run1.mp4** : Successful Video on run on track1

### Output videos:
** run1.mp4 **

Initially I tried to replicate the clone.py in the tutorial which used LeNet in the beginning. This can be found at [clone.py](clone.py)
Next, I tried to replace the clone.py model to nVidia Autonomous Car Group Model and alongside adding the data generator function. This can be found at [model.py](model.py)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Initially, my approach was to get a working condition model for the simulator and I used LeNet for a simple model. But the car wouldn't stay in the lane for long even on the basic track. So, I followed with the nVidia Model which was able to succesfully run on the basic track.

A model summary is as follows:

```
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_2[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
conv2d_1 (Conv2d)  				 (None, 43, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
conv2d_2 (Conv2d)  				 (None, 20, 77, 36)    21636       conv2d_1[0][0]            
____________________________________________________________________________________________________
conv2d_3 (Conv2d)  				 (None, 8, 37, 48)     43248       conv2d_2[0][0]            
____________________________________________________________________________________________________
conv2d_4 (Conv2d)  				 (None, 6, 35, 64)     27712       conv2d_3[0][0]            
____________________________________________________________________________________________________
conv2d_5 (Conv2d)  				 (None, 4, 33, 64)     36928       conv2d_4[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)  			 (None, 4, 33, 64)     36928       conv2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 8448)          0           conv2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           844900      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
```

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 98). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 90-105). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually ([model.py line 146](model.py#104)).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

NOTE: I have a `preprocessing.py` file which can create lots of augmented data based on brightness, shadow, cropping, etc. But, due to the slow speed of the current environment, I tried training my model on aws image that I had. But, due to version mismatch the trained model didn't work in the udacity gpu workspace. I posted on Student Hub but didn't find a good solution. Thus, I went ahead with just flipping the image which was enough for a successful first track run.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to try LeNet model with 5 epochs using the Udacity training data. On the first track, it ran well for 1/4th the track before plunging into the lake. After doing some preprocessing (Flipping), it was able to avoid the lake but was too wobbly.

Next, I changed the model to nVidia model which successfully was able to run the entire track. It is at this point that the centre, left and right images were joined to increase the accruacy of the prediction.

At the end of the process, the vehicle is able to drive autonomously around the track 1 without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 90-105) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![Final model architecture](images/nVidia_model.png)

#### 3. Creation of the Training Set & Training Process

I used the Udacity sample data as the GPU mode was too slow for me in order to both create data and train model on augmented data.


Challenges:

1. The greates challenge for me was finding the best model and obtaining accuracy in the track2.
2. Due to slow gpu and technical difficulties, I couldn't run the model trained on aws. Student Hub wasn't able to provide a workaround, so had to avoid data augmentation (time taken to train last model: s)
3. I created some preprocessing steps and it was difficult to find which ones would provide the best results.

Future Enhancements: 

1. Creating a highly complex model which can generalize all important features. 
2. Creating more preprocessing steps to improve data augmentation.

For Mentor: I have added my aws video in which I display a successful track2 run. This was due to my aws trained model not being able to run in udacity environment. Student Hub provided a solution but it didn't work due to library mismatch and when I tried to replicate, got a segmentation error which i couldn't solve. Please accept the track2 video as final submission for track2 video.mp4