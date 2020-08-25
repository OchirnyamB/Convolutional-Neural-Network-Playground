# Convolutional-Neural-Network-Playground
Implementing CNN network architectures with Keras library

### Requirements:
* python3
* keras
* numpy V1.19.2
* scikit-image V0.17.2
* opencv-python
* matplotlib V3.2.2
* imutils 
* jupyter

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

3. **MiniVGGNet** network strucutre:
> CONV => RELU => CONV => RELU => POOL x2 FC=>RELU=>FC=>SOFTMAX

Layer Type | Output Size | Filter Size / Stride
---------- | ------------| -----------
INPUT IMAGE | 32x32x3 |
CONV | 32x32x3 | 3x3, K=32
ACT  | 32x32x32 | 
BN | 32x32x32 | 
CONV | 32x32x32 | 3x3, K=32
ACT  | 32x32x32 | 
BN | 32x32x32 | 
POOL | 16x16x32 | 2x2
DROPOUT | 16x16x32 | 
CONV | 16x16x64 | 3x3, K=64
ACT  | 16x16x64 | 
BN | 16x16x64 | 
CONV | 16x16x64 | 3x3, K=64
ACT  | 16x16x64 |
BN   | 16x16x64 |
POOL | 8x8x64   | 2x2
FC   | 512 |
ACT  | 512 |
BN   | 512 |
DROPOUT | 512 |
FC      | 10 |
SOFTMSX | 1O |

4. **ResNet50** network model:
> Aimed at labeling and categorizing images into almost 22,000 categories
> Follows the WordNet hierarchy. Each meaningful word/phrase inside WordNet is called a synonym set.
> 50 weight layers, deeper than both VGG16 and VGG19

### MNIST Dataset:
> "Hello World" equivalent of deep learning applied to image classification
> The **MNIST** (“NIST” stands for National Institute of Standards and Technology while “M” stands for “modified” as the data has been preprocessed to reduce any burden on Computer Vision processing and focus solely on the task of **digit recognition**).

> Dataset consisting of 70,000 data points (7,000 examples per digit). Each data point is represented by a 784-d vector (flattened 28x28 images).

### CIFAR-10 Dataset:
> Dataset consisting of 60,000 (32x32x3 RGB images) resulting in a feature vector dimensionality of 3072. It consists of 10 classes: _airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks_.

### Flowers-17 Dataset:
> Dataset consisting of 1360 images, 17 distinct images species of flowers. 

### Evaluations of the Trained Networks:
* MNIST **LeNet** Classification: [99% Accuracy on average](output/lenet_mnist_trainingEval.txt
)
![kerasCIFAR10](/output/lenet_mnist.PNG)

* CIFAR-10 **ShallowNet** Classification: [60% Accuracy on average](output/shallownet_cifar10_trainingEval.txt)
![kerasCIFAR10](/output/shallownet_cifar10.PNG)
* CIFAR-10 **VGGNet** Classification: [81% Accuracy on average](output/minivggnet_cifar10_trainingEval.txt)
![kerasCIFAR10](/output/minivggnet_cifar10.PNG)
* Flowers-17 
    * With Data Augmentation: 
       * [71% Accuracy on average](output/minivggnet_flowers17_data_aug.txt)
           ![minivggf17](/output/minivggnet_flowers17_data_aug.PNG)
    * With Data Augmentation & Fine Tuning with VGG16: 
       * [95% Accuracy on average](output/finetune_flowers17.txt)

* Image Classification using pretrained ResNet50 from Keras:
[pretrainedResNet50](./classifyImage_ResNet.ipynb)
### References:
* Deep Learning for Computer Vision with Python VOL1 & VOL2 by Dr.Adrian Rosebrock



