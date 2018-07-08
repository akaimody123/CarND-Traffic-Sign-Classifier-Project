# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./1.png "Traffic Sign 1"
[image5]: ./2.png "Traffic Sign 2"
[image6]: ./3.png "Traffic Sign 3"
[image7]: ./4.png "Traffic Sign 4"
[image8]: ./5.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed based on classes

![alt text][image1]
 
### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data in order to reduce bia on images with bigger data.(normlized using (pixel-128)/128)

I decided to generate additional data because there is a imbalance in the data distribution. Some classes has more than 2000 images while some only has 200. I used image generator to generate more images(eg. some rotation based on orginal image is recognised as same class)  for thoses classes with images less than 1000.
 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 

rotation_range=15., zoom_range=0.2, width_shift_range=0.1, height_shift_range=0.1

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x32 				|
| Dropout	         	| Keep 0.9                                      |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 13x13x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x64 				    |
| Dropout	         	| Keep 0.9                                      |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 4x4x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x128 				    |
| Dropout	         	| Keep 0.9                                      |
| Fully connected		| Input 512, output 120     					|
| RELU				    |             									|
| Dropout	         	| Keep 0.6                                      |
| Fully connected		| Input 120, output 84     					    |
| RELU				    |             									|
| Dropout	         	| Keep 0.6                                      |
| Fully connected		| Input 84, output 43     					    |
| RELU				    |             									|
| Dropout	         	| Keep 0.6                                      |
|						|												| 
|						|												|
 
I tried with color images and grayscale (both normalized), it seems there is no big difference, so i go with the color ones.  There are three CNN layers and three layers of fully connected NN. As this model is overfitted, i used dropout. I assume tunning the dropout could get better result.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used following parameters
EPOCH:100
BATCH SIZE:512
LEARNING RATE:0.001
DROP_RATE=[0.9,0.6]



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.964
* test set accuracy of 0.948

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

First architecture was LeNet solution. It showed poor results. Then i tried to increase epoch and batch sizes as well as changing the architecture such as increasing CNN channels and change the size of filters. It helped improve the accuracy to about 0.9. But it seems the model is overfitted( as the validation accuracy does not really increase as epochs increases.) So i added dropout and tuned a bit the drop rate.

If a well known architecture was chosen:
* What architecture was chosen? LeNet
* Why did you believe it would be relevant to the traffic sign application? Because traffic signs and hand written letters are all images with certain patterns.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
All accuracy are quite high.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| speed limit(60km/h)   | speed limit(60km/h)   						| 
| No entry     			| No entry 										|
| Road work				| Road work										|
| Bumpy road	      	| Bumpy Road					 			    |
| Turn right ahead		| Turn right ahead      						|

Test Accuracy =  1.0
TOp5 =  TopKV2(values=array([[9.8407078e-01, 1.5906993e-02, 1.2829579e-05, 4.6933146e-06,
        3.4256300e-06],
       [1.0000000e+00, 1.2716619e-10, 1.3073855e-13, 1.3625431e-15,
        1.1934553e-16],
       [1.0000000e+00, 2.1700940e-08, 2.6106821e-12, 1.0075693e-13,
        2.5733979e-14],
       [9.9999964e-01, 3.2166935e-07, 7.7999736e-08, 5.0443161e-09,
        3.5541461e-09],
       [1.0000000e+00, 8.7398053e-09, 1.1175476e-09, 1.1149708e-10,
        5.7840503e-11]], dtype=float32), indices=array([[ 3,  5, 16, 30,  6],
       [17, 22, 29,  0, 26],
       [25, 20, 31, 11, 26],
       [22, 31, 29, 19, 26],
       [33, 34, 36, 40, 35]], dtype=int32))
Correct Predicted =  [ True  True  True  True  True]
predicted label =  [ 3 17 25 22 33]


The model was able to corectly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Please see above

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were (see before)

Top 1 probability

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .984         			| speed limit(60km/h)   						| 
| 1.0     			    | No entry 										|
| 1.0				    | Road work										|
| 1.0	      	        | Bumpy Road					 			    |
| 1.0		            | Turn right ahead      						|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


