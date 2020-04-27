import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F

from Autoencoders.layers import Flatten


class Encoder2DConv(nn.Module):
    """Constructs an encoder for use in various autoencoder models.
    TODO: Consider passing a list to designate the number of convolutional
          filters per layer.
    TODO: Create another class for dilated convolutions and
          causal dilated convolutions.
    TODO: Add blocks and residuals? - Maybe better for a different class.

    Args:
        latentdims (int):        Number of dimensions in the latent space
        nchannels (int):         Number of channels in the input data.
                                 Default = 1.
        nfilters (int):          Number of filters in each layer.
                                 Default is 32.
    """
    def __init__(
        self,
        inputdims,
        latentdims,
        nlayers=2,
        nchannels=1,
        nfilters=32,
        kernel_size=3,
        stride=1,
        padding=1,
        use_batchnorm=False
    ):

        super(Encoder2DConv, self).__init__()

        #arguments to Conv2D:
        # in_channels, out_channels, kernel_size,
        # stride, padding, dilation, groups, bias,
        # padding-mode
        self.nchannels = nchannels
        self.kernel_size = kernel_size
        self.stride = stride
        self.inputdims = inputdims

        # string together arbitrary number of convolutional layers
        convlayers = []
        for layer in range(nlayers):
            if layer == 0:
                #first layer, in_channels = nchannels
                convlayers.append(nn.Conv2d(nchannels, nfilters, kernel_size, stride, padding))
                if use_batchnorm:
                    convlayers.append(nn.BatchNorm2d(nfilters))
                convlayers.append(nn.ReLU())

            else:
                convlayers.append(nn.Conv2d(nfilters, nfilters, kernel_size, stride, padding))
                convlayers.append(nn.ReLU())
        self.convlayers = nn.Sequential(*convlayers)

        self.flatten = Flatten()
        self.latent = nn.Linear(nfilters*inputdims[0]*inputdims[1], latentdims)

    def forward(self, x):
        x = self.convlayers(x)
        x = self.flatten(x)
        return self.latent(x)


class VAEEncoder2DConv(nn.Module):
    """Constructs an encoder for use in various autoencoder models.
    TODO: Consider passing a list to designate the number of convolutional
          filters per layer.
    TODO: Create another class for dilated convolutions and
          causal dilated convolutions.
    TODO: Add blocks and residuals? - Maybe better for a different class.

    Args:
        latentdims (int):        Number of dimensions in the latent space
        nchannels (int):         Number of channels in the input data.
                                 Default = 1.
        nfilters (int):          Number of filters in each layer.
                                 Default is 32.
    """
    def __init__(
        self,
        inputdims,
        latentdims,
        nlayers=2,
        nchannels=1,
        nfilters=32,
        kernel_size=3,
        stride=1,
        padding=1,
        use_batchnorm=False
    ):

        super(VAEEncoder2DConv, self).__init__()

        #arguments to Conv2D:
        # in_channels, out_channels, kernel_size,
        # stride, padding, dilation, groups, bias,
        # padding-mode
        self.nchannels = nchannels
        self.kernel_size = kernel_size
        self.stride = stride
        self.inputdims = inputdims

        # string together arbitrary number of convolutional layers
        convlayers = []
        for layer in range(nlayers):
            if layer == 0:
                #first layer, in_channels = nchannels
                convlayers.append(nn.Conv2d(nchannels, nfilters, kernel_size, stride, padding))
                if use_batchnorm:
                    convlayers.append(nn.BatchNorm2d(nfilters))
                convlayers.append(nn.ReLU())

            else:
                convlayers.append(nn.Conv2d(nfilters, nfilters, kernel_size, stride, padding))
                convlayers.append(nn.ReLU())
        self.convlayers = nn.Sequential(*convlayers)

        self.flatten = Flatten()
        self.mu = nn.Linear(nfilters*inputdims[0]*inputdims[1], latentdims)
        self.logvar = nn.Linear(nfilters*inputdims[0]*inputdims[1], latentdims)

    def forward(self, x):
        x = self.convlayers(x)
        x = self.flatten(x)
        return self.mu(x), self.logvar(x)