import torch
import numpy as np
from torchvision import transforms, models
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
#% matplotlib inline

# Image style transfer using convolutional neural networks

# loading in the pre-trained VGG19 model features with paramaters frozen
from load_model import load_model
vgg = load_model()

# move the model to GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

# load in content and style images
from load_image import load_image

content = load_image('images/content/pic1.jpg').to(device)
# resize style to match content
style = load_image('images/style/pic2.jpg', shape=content.shape[-2:]).to(device)

# display the images
from im_convert import im_convert
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20,10))
#content and style images shown side by side
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(style))

# getting the content and style features before forming the target image
from get_features import get_features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# calculating the gram matrices for each layer of the style representation
from gram_matrix import gram_matrix
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# creating the target image and prepping it for change
# to start of, we use a copy of our content image as the initial target and then iteratively change its style
target = content.clone().requires_grad_(True).to(device)

# setting the weights for each style layer and setting content and style weights
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

content_weight = 1 # alpha
style_weight = 1e6 # beta

# updating the target and calculating losses
from update_loss import update_loss
update_loss(target, vgg, content_features, style_weights, style_grams, content_weight, style_weight)

# display the target image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(target))
