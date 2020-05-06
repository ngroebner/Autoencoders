# Various autoencoders for seismic analysis

Written in Pytorch.

Library for building autoencoders.  Various encoder and decoder layers, latent layers, clustering layers.

To install, clone the repository and type

`pip install .`

So far have 2D convolutional regular and variational autoencoders done.

To do:

1. Causal convolutional encoders and decoders for spectrograms (2D) and seismograms (1D)
2. [Convolutional temporal encoder/decoder](https://arxiv.org/pdf/1806.09174.pdf). 
3. [Deep convolutional embedded clustering.](https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf)
4. Train encoders together, decoders separately based on basin, source, etc to learn generalized encodings of seismic sources. Like [universal music translation network](https://arxiv.org/abs/1805.07848) / DeepFakes.
