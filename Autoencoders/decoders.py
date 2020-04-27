import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F

from Autoencoders.layers import Flatten, UnFlatten

class Decoder2DConv(nn.Module):
    """Constructs an decoder for use in various autoencoder models.
    This is antisymmetric to the Encoder2DConv class.  I.e., it
    takes as input a latent vector and outputs a 2D matrix with
    dimensions outputdims.

    TODO: Add parameter to define number of convolutional layers.
    TODO: Add blocks and residuals? - Maybe better for a different class.
    Args:
        latentdims (int):        Number of dimensions in the latent space
        nchannels (int):         Number of channels in the original input data.
                                 Default = 1.
        nfilters (int):          Number of filters in each layer of the encoder.
                                 Default is 32.

    """
    def __init__(
        self,
        outputdims,
        latentdims,
        nlayers=2,
        nchannels=1,
        nfilters=32,
        kernel_size=3,
        stride=1,
        padding=1,
        use_batchnorm=False
    ):

        super(Decoder2DConv, self).__init__()

        self.nchannels = nchannels
        self.kernel_size = 3
        self.stride = 1
        self.outputdims = outputdims
        self.nfilters = nfilters

        self.latentin = nn.Linear(latentdims, nfilters*outputdims[0]*outputdims[1])
        self.unflatten = UnFlatten()

        # string together arbitrary number of convolutional layers
        convlayers = []
        for layer in range(nlayers):
            if layer == nlayers - 1:
                #last layer, out_channels = nchannels, sigmoid activation layer
                convlayers.append(nn.Conv2d(nfilters, nchannels, kernel_size, stride, padding))
                convlayers.append(nn.Sigmoid())
            else:
                convlayers.append(nn.Conv2d(nfilters, nfilters, kernel_size, stride, padding))
                if use_batchnorm:
                    convlayers.append(nn.BatchNorm2d(nfilters))
                convlayers.append(nn.ReLU())
        self.convlayers = nn.Sequential(*convlayers)

    def forward(self, x):
        x = self.latentin(x)
        x = self.unflatten(x, self.nfilters, self.outputdims)
        return self.convlayers(x)

