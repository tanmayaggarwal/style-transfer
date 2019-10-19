import torch
import numpy as np
from torchvision import transforms, models
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from get_features import get_features
from gram_matrix import gram_matrix
from im_convert import im_convert
#% matplotlib inline

# function to update the target and calculate losses

def update_loss(target, vgg, content_features, style_weights, style_grams, content_weight, style_weight):
    # for displaying the target image intermittently
    show_every = 400

    # iteration hyperparamaters
    optimizer = optim.Adam([target], lr=0.004)
    steps = 2000            # variable; can be updated as needed

    for ii in range(1, steps+1):
        target_features = get_features(target, vgg)

        # calculate the content loss
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        # calculate the style loss iterating through a number of layers
        style_loss = 0
        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            batch_size, d, h, w = target_feature.shape
            # get the "style" style representation for the layer
            style_gram = style_grams[layer]
            # the style loss for one layer weighted
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            # add to the style loss
            style_loss += layer_style_loss / (d * h * w)

        # calculate the total loss
        total_loss = content_weight * content_loss + style_weight * style_loss

        # update the target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # display intermediate images and print the loss
        if ii % show_every == 0:
            print('Total loss: ', total_loss.item())
            plt.imshow(im_convert(target))
            plt.show()
