# Convolutional-Neural-Network-Playground
Implementing CNN network architectures with Keras library

### Requirements:
* python3
* keras
* numpy V1.19.2
* scikit-image V0.17.2
* opencv-python
* matplotlib V3.2.2

### CNN architectures implemented in this repository:
1. **ShallowNet** network structure:
> INPUT => CONV => RELU => FC
2. **LeNet** network structure:
>  INPUT => CONV => TANH => POOL => CONV => 
>  TANH => POOL => FC => TANH => FC 
    
Layer Type | Output Size | Filter Size / Stride
---------- | ------------| -----------
INPUT IMAGE | 28x28x1 |
CONV | 28x28x20 | 5x5, K=20
ACT  | 28x28x20 |
POOL | 14x14x20 | 2x2
CONV | 14x14x50 | 5x5, K=50
ACT  | 14x14x50 |
POOL | 7x7x50 | 2x2
FC   | 500 |
ACT  | 500 |
FC   | 10 |
SOFTMAX | 10 |

### MNIST Dataset:
> "Hello World" equivalent of deep learning applied to image classification
> The **MNIST** (“NIST” stands for National Institute of Standards and Technology while “M” stands for “modified” as the data has been preprocessed to reduce any burden on Computer Vision processing and focus solely on the task of **digit recognition**).

> Dataset consisting of 70,000 data points (7,000 examples per digit). Each data point is represented by a 784-d vector (flattened 28x28 images).

### CIFAR-10 Dataset:
> Dataset consisting of 60,000 (32x32x3 RGB images) resulting in a feature vector dimensionality of 3072. It consists of 10 classes: _airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks_.

### Evaluations of the Trained Networks:

* CIFAR-10 Classification: [60% Accuracy on average](output/shallownet_cifar10_trainingEval.txt
)
![kerasCIFAR10](/output/shallownet_cifar10.PNG)

### References:
* Deep Learning for Computer Vision with Python VOL1 by Dr.Adrian Rosebrock



