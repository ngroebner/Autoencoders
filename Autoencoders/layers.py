



class Flatten(nn.Module):
    """Flattens a convolutional layer into linear layer.
    """
    def forward(self, x):
        return x.view(x.size(0), -1)

class UnFlatten(nn.Module):
    """Takes a flattened layer and reconstructs it 
    into a multidimensional layer.
    """
    def forward(self, x, channels, size):
        return x.view(x.size(0), channels, *size)

def CausalConv1D(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
    pad = (kernel_size - 1) * dilation
    return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)

def CausalConvTranspose1D(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
    pad = (kernel_size - 1) * dilation
    return nn.ConvTranspose1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)

