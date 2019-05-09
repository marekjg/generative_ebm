# Image synthesis with energy based models

Simple PyTorch implementation of [Implicit Generation and Generalization in Energy Based Models](https://arxiv.org/pdf/1903.08689.pdf). The images are generated by the descent on the energy function using Langevin stochastic gradient descent.

Samples from the conditional energy model (low quality in the bottom row comes from the fact that the last row is started from uniform noise and the rest of the samples were taken from replay buffer):

![Conditional energy model](cond_mnist.png)
