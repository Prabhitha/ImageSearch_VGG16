## Image search engine using Deep Learning Model based on matsui528 code.

Just like Google image search, here we are going to build a simple Image search engine using the Deep learning model VGG16. Given a query image, we are going to display images similar to it. We are going to implement it using Keras (It is an open-source software library that provides a Python interface for artificial neural networks)and Flask (It is a micro web framework written in Python).

### Table of Contents:
1. VGG16 Architecture<br>
2. Image search engine flow and code<br>
3. Conclusion<br>

### Example:

![image](https://github.com/Prabhitha/ImageSearch/blob/master/applesOutput.png)

### 1. VGG16 Architecture:

VGG16 is a convolutional neural network model that is 16 years deep. It is trained on the ImageNet dataset. The input image should always has the size of 224x224x3 for the VGG16 to work. Here, we are not going to train the VGG16 from scratch because it will take large amount of time and GPU utilization. So we are going to use the weights of the pre-trained model.

VGG16 has multiple convolution, max pool and full connected layers.
**Convolution**: In this layer, we perform the convolution operation over the image using a filter/kernel to detect the edges in the image.
**Max-pool:** It is used to reduce the spatial size of the image and it also helps to reduce over-fitting.

### 2. Image search engine flow and code:

To implement our image search engine, we are going to use the images from COCO dataset to train our model.

### Steps to Implement:
* First, we will convert our images into the size 224x224x3 because the input of VGG16 should always be 224x224x3.<br>
* Next, we are going to extract the deep features from all the images in the training dataset (i.e. we are going to convert the image into vector). To extract the deep features, we are passing our input images through the 16 layered VGG model. In the final layer (16th layer) of VGG model, we have 1000 parameters which is used to classify the input images. In our example we will extract the parameters/features from the 15th layers which is a Fully connected layer with 4096 parameters.<br>
* Basically, we are just converting all our input images of the training dataset into a vector of size 4096 with the help of VGG16.<br>
* We will then convert the data to an n-dimensional array and perform data pre-processing.<br>

Now we have converted the images into features, our next step is, given an image, we need to find the images which are similar to query image.

* To do that, whenever we get a query image, we have to convert the query image to vectors by using the same feature extraction technique which we applied on the training images.<br>
* Then, to identify the similarity between the query image and training data, we calculate the Euclidean distance between the query image and all the training images and sort them based on scores. Euclidean distance/ L2 Norm is the shortest distance between two points in an N dimensional space. It is used as a common metric to measure the similarity between two data points.<br>

To get the query input image and to display the output images to the website, we are using Flask API. Below is the simple html code in order to perform this function.

### 3. Conclusion:
With the help of VGG16, we have trained a simple image search engine and by training the model on a large dataset, we can have high accuracy.

### Note:
This has been added to my Medium story in the below link. <br>
https://prabhitha3.medium.com/image-search-engine-using-deep-learning-model-c452d2637cf6
