import torch
import numpy as np
from torchvision import transforms, models
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
#% matplotlib inline

# calculate the Gram Matrix of a tensor

def gram_matrix(tensor):
    # get the batch_size, depth, height, and width of a tensor
    # reshape it, so we are multiplying the features for each channel
    # calculate the gram matrix

    batch_size, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)

    gram = torch.mm(tensor, tensor.t())

    return gram
