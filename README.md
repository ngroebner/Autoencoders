# Various autoencoders for seismic analysis

Written in Pytorch.

Library for building autoencoders.  Various encoder and decoder layers, latent layers, clustering layers.

To install, clone the repository and type

`pip install .`

So far have 2D convolutional regular and variational autoencoders done.

To do:

1. Causal convolutional encoders and decoders for spectrograms (2D) and seismograms (1D)
2. [https://arxiv.org/pdf/1806.09174.pdf](Convolutional temporal encoder/decoder). 
3. [https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf](Deep convolutional embedded clustering.)
4. Train encoders together, decoders separately based on basin, source, etc to learn generalized encodings of seismic sources. Like [https://arxiv.org/abs/1805.07848](universal music translation network) / DeepFakes.
