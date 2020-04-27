# Example training a variational autoencoder

from torch import nn, optim
from Autoencoders.encoders import VAEEncoder2DConv
from Autoencoders.decoders import Decoder2DConv
from Autoencoders.autoencoders import VAE
from Autoencoders.losses import vae_loss
