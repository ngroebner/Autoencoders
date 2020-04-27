from torch import nn


class Autoencoder(nn.Module):
    """Constructs an autoencoder given an encodermodule, decoder module.
    Think about whether to make this an abstract class. And whether to
    incorporate the loss function.
    """
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)