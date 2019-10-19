import torch
import numpy as np
from torchvision import transforms, models
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
#% matplotlib inline

# pre-trained VGG19 model features with paramaters frozen
def load_model():
    vgg = models.vgg19(pretrained=True).features

    # freeze all VGG parameters since we are only optimizing the target Image
    for param in vgg.parameters():
        param.requires_grad_(False)

    return vgg
