import torch
from torch import nn
import torch.nn.functional as F

# from pytorch examples
# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss(recon_x, x, mu, logvar):
    MSE = F.binary_cross_entropy_loss(recon_x, x) #, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())/)
    KLD /= x.view(-1, input_size).data.shape[0]

    return MSE + KLD, MSE, KLD
