from torch import nn


class Autoencoder(nn.Module):
    """Constructs an autoencoder given an encoder module, decoder module.
    Think about whether to make this an abstract class. And whether to
    incorporate the loss function.

    Args:
        encoder (Encoder):      Pytorch module implementing the encoder
        decoder (Decoder):      Pytorch module implementing the decoder

    The input size to the encoder and the output size from the decoder
    must match.

    The output of the encoder (latent space) must match the
    input to the decoder
    """
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # implement dimension checking here

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


class VAE(nn.Module):
    """Constructs a variational autoencoder given an encoder module, decoder module.
    Think about whether to make this an abstract class. And whether to
    incorporate the loss function.

    Args:
        encoder (Encoder):      Pytorch module implementing the encoder
        decoder (Decoder):      Pytorch module implementing the decoder

    The input size to the encoder and the output size from the decoder
    must match.

    The encoder must output two latnet vectors, mu and logvar, each of
    dimension latentdim.  The decoder must take an inpout of size latentdim.
    """
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar