# Convnext-based Alzheimer's Classifier for the ADNI dataset
Author: Feiyang Shan
## Project Overview
This project aims to use a lastest model to identify Alzheimer’s disease (AD) using the MRI images in the ADNI dataset(Project 8). Our goal is to reach an accuracy of the 80% on the testset.
The project uses the ConvNeXt 

## ConvNeXt Introduction
ConvNeXt is a modern model architecture based on convolutional neural networks (CNNs), proposed by Facebook AI Research. Its design draws inspiration from new architectures such as the Vision Transformer (ViT), while optimizing and improving traditional convolutional networks to achieve performance comparable to that of the Transformer model while maintaining the efficiency and ease of use of convolutional networks.

### ConvNeXt Stucture
#### Downsampling Layers
The ConvNeXt class begins with a series of downsampling layers, which progressively reduce the spatial resolution of the input image while increasing the feature dimensions. The first layer, referred to as the "stem," uses a convolutional layer with a stride of 4 to downsample the input image.
```python
stem = nn.Sequential(
    nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
    LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
)
```
- Purpose: The stem layer reduces the input image size by a factor of 4 in both height and width, while projecting the input channels into the first feature dimension.
- LayerNorm: Normalization is applied to stabilize training and improve convergence.
Subsequent downsampling layers use a combination of LayerNorm and convolutional layers to further reduce the spatial dimensions while increasing the feature dimensions.

#### ConvNeXt Blocks
The ConvNeXtBlock class represents the core building block of the ConvNeXt architecture. Each block consists of the following components:
- Depthwise Convolution:
``` python
self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
``` 
- LayerNorm:
```python
self.norm = LayerNorm(dim, eps=1e-6)
```
    - Normalization is applied in the "channels_last" format to improve computational efficiency.
- Pointwise Convolutions
```python
self.pwconv1 = nn.Linear(dim, 4 * dim)
self.pwconv2 = nn.Linear(4 * dim, dim)
```
    - Two pointwise convolutions (implemented as linear layers) are used to expand and then reduce the feature dimensions, enabling the block to learn complex feature representations.
- Activation and Residual Connection:
    - A GELU activation function is applied between the pointwise convolutions.
    - A residual connection is added to stabilize training:
```
x = input + self.drop_path(x)
```

#### Classification Head
The classification head consists of a global average pooling layer followed by a fully connected layer:
```python
self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
self.head = nn.Linear(dims[-1], num_classes)
```
- Global Average Pooling: The spatial dimensions are averaged to produce a single feature vector for each image.
- Fully Connected Layer: The feature vector is mapped to the number of output classes (e.g., 2 for AD vs. NC).

#### DropPath Regularization
The DropPath class implements stochastic depth, a regularization technique that randomly skips certain blocks during training
```
self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
```
- Reduces overfitting by allowing the model to learn more robust features.
#### Predefined Configurations
The module provides several predefined configurations for different model sizes:
ConvNeXt-Tiny:  
```python
depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]  
ConvNeXt-Small:  
depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]  
ConvNeXt-Base:  
depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]  
```

## Dataset Structure
The ADNI dataset follows this structure:
```
AD_NC/
├── train/
│   ├── AD/
│   └── NC/
└── test/
    ├── AD/
    └── NC/
```
In `dataset.py`, the training folder is split into training and validation sets. The split is done randomly, with 80% of the training folder allocated to the training set and the remaining 20% allocated to the validation set at the start of each training session. In the training or validation set, MRI slices from the same patient will not appear in both the training and validation sets simultaneously to avoid data leakage caused by similar images.

### data preprocessing
`dataset.py` have several functions, incluing mean and std calculation, data pre-processing and datastructure check.


### Command to Run data check
``` 
python dataset.py
```

## Training Process
### Training methods used
- Warmup
- Drop path (similar to drop out)
- Cosine annealing learning rate
- Mixup
- EMA
- Gradient Clipping

### Hyperparameters
```
BATCH_SIZE = 32
EPOCHS = 100  
BASE_LR = 3e-4 
MIN_LR = 5e-5 
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 0.2
LABEL_SMOOTHING = 0.1 
DROP_PATH_RATE = 0.6 
GRAD_CLIP_NORM = 10.0 
USE_EMA = True
SEED = 42
EVAL_WITH_EMA = True
USE_MIXUP = False
MIXUP_ALPHA = 0.2
```
### Training Results


## Predict Results
After running, the predictions of 5 randomly selected pictures from the test set and the chaos matrix of all MRI pictures will be displayed.

## Dependencies
To reproduce this project, install the following dependencies:
- python 3.13.7
- pytorch: 2.8.1
- matplotlib: 3.10.6
- cuda:12.8
- numpy: 2.3.2
- scikit-learn: 1.7.1

## Conclution
The ConvNeXt-based classifier achieved a test accuracy of 76% on the ADNI dataset. This demonstrates the effectiveness of modern convolutional architectures in medical image classification tasks, providing a solid foundation for further research and optimization.
## References
Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. arXiv. https://arxiv.org/abs/2201.03545
