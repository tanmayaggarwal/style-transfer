# style-transfer
Style transfer with deep neural networks

<main file> style_transfer.py

This is a style transfer application that uses a pretrained 19-layer VGG network. The VGG network is used to extract content and style features from input images.

User is encouraged to use a style and content image of their own (see below for input images path).

Input images path:
Content: /images/content/pic1.jpg
Style: /images/content/pic2.jpg

Dependencies:

%matplotlib inline

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models

Default values:
Number of steps to update target image: 2000
Content_weight / Style_weight ratio: 1 / 1e6

Content loss: MSE loss
Style loss: Gram matrix (calculated for each layer to result in layer_style_loss)

Note: application is compatible with CUDA if available (recommended to reduce execution time period)