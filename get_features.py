import torch
import numpy as np
from torchvision import transforms, models
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
#% matplotlib inline

# function to get content and style features
def get_features(image, model, layers=None):
    ''' Run an image forward through a model and get the features for a set of layers. Default layers are for VGGNet matching Gatys et al (2016) '''

    # Mapping layer names of VGGNet to names from the paper
    if layers is None:
        layers = {'0':'conv1_1',
                  '5':'conv2_1',
                  '10':'conv3_1',
                  '19':'conv4_1',
                  '21':'conv4_2', ## content representation
                  '28':'conv5_1',
                  }

    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features
