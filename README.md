# PyTorch-Course-2019
My course work from the following Udemy course: https://www.udemy.com/pytorch-for-deep-learning-and-computer-vision/


This course covered a lot of material pertaining to PyTorch and computer vision applications. The notebooks in this repository increse in complexity in ascending order.


It is important to note that the code in the following notebooks can be made much cleaner through refactoring. When writing this code I really wanted to emphasize the progression of my learning and understanding of the material. Therefore, some of the code is not PEP 8 compliant.

If you are having issues viewing a notebook in GitHub's native viewer please copy the link to the notebook into the following website: https://nbviewer.jupyter.org/


## Notebook #1 - Tensor Operations

The purpose of this notebook was to learn/practice basic tensor operations in PyTorch.


## Notebook #2 - Linear Regression Model

The purpose of this notebook was to implement (and visualize) a linear regresssion model in PyTorch.


## Notebook #3 - Perceptron

The purpose of this notebook was to implement a basic perceptron model in PyTorch.

A perceptron is simply a single layer neutral network that takes in multiple input values (or one input layer) and outputs a single output corresponding to one of two classes (it is a binary classifier). 

Perceptron process:

1. It takes multiple input values
2. It multiplies those input values by their respective weights (and adds a bias value). The weights and biases are decided earlier through training.
3. Then, it linearly combines (adds) these values, this results in a single scalar value
4. Then it takes this single value and inputs it into an activation function (such as the sigmoid function)
5. Finally it outputs this result. This result is the model's prediction regarding which of the two classes the input falls into. 


## Notebook #4 - Deep Neural Network

The purpose of this notebook was to implement a deep neural network in PyTorch to classify a more complex dataset. A deep neutral network is a model with n number of hidden layers (in this case there are 4 hidden layers).


## Notebook #5 - Image Classification (MNIST)

The purpose of this notebook was to create a model (using linear layers) that can, with high accuracy (~96%), classify the handwritten digits in the MNIST dataset. This notebook also allowed me to use PyTorch's image transformations and various other helper functions.


## Notebook #6 - CNN (MNIST Classification)

The purpose of this notebook was to classify between handwritten digits in the MNIST dataset using a convolutional neural network (CNN). Using a CNN we can achieve extremely high validation accuracy (~99%) due to its excellent feature extraction.  


## Notebook #7 - CNN (CIFAR10 Classification)

The purpose of this notebook was to show the LeNet model used for the CIFAR10 dataset. Additionally, I wanted to show how tuning hyperparmeters can increase the accuracy and prevent overfitting. Obviously, better results can be achieved using different model architectures, but the purpose of this notebook was not to achive state-of-the-art results. It was to show a progression of improvement.


## Notebook #8 - Transfer Learning

The purpose of this notebook was to classify between pictures of ants and bees using transfer learning from the VGG16 pretrained model.


## Notebook #9 - Style Transfer

The purpose of this notebook was to implement style transfer in PyTorch using a neural network pretrained on the VGG19 model.

Style transfer is the task of taking two images, a "content image" and a "style image", and generating a new photo ("target image") that uses the content from the "content image" and the style from the "style image".
