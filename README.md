# style-transfer
Style transfer with deep neural networks

<main file> style_transfer.py

This is a style transfer application that uses a pretrained 19-layer VGG network. The VGG network is used to extract content and style features from input images.

User is encouraged to use a style and content image of their own (see below for input images path).

Input images path:
Content: /images/content/pic1.jpg <br/>
Style: /images/content/pic2.jpg <br/>

Dependencies:

%matplotlib inline <br/>
from PIL import Image <br/>
import matplotlib.pyplot as plt <br/>
import numpy as np <br/>
import torch <br/>
import torch.optim as optim <br/>
from torchvision import transforms, models <br/>

Default values: <br/>
Number of steps to update target image: 2000 <br/>
Content_weight / Style_weight ratio: 1 / 1e6 <br/>

Content loss: MSE loss <br/>
Style loss: Gram matrix (calculated for each layer to result in layer_style_loss) <br/>

Note: application is compatible with CUDA if available (recommended to reduce execution time period)
