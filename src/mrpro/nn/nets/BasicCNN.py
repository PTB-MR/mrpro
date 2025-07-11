from collections.abc import Sequence
from itertools import pairwise

import torch
from torch.nn import ReLU

from mrpro.nn.FiLM import FiLM
from mrpro.nn.ndmodules import BatchNormND, ConvND
from mrpro.nn.Sequential import Sequential


class BasicCNN(Sequential):
    def __init__(
        self,
        dim: int,
        channels_in: int,
        channels_out: int,
        batch_norm: bool = True,
        n_features: Sequence[int] = (64, 64, 64),
        cond_dim: int = 0,
    ):
        """Initialize a  basic CNN.

        Parameters
        ----------
        dim
            The number of spatial dimensions of the input tensor.
        channels_in
            The number of input channels.
        channels_out
            The number of output channels.
        batch_norm
            Whether to use batch normalization.
        n_features
            The number of features in the hidden layers. The length of this sequence determines the number of hidden layers.
        cond_dim
            The dimension of the condition tensor. If 0, no FiLM conditioning is applied.
        """
        super().__init__()
        use_film = cond_dim > 0
        self.append(ConvND(dim)(channels_in, n_features[0], kernel_size=3, padding='same'))
        for c_in, c_out in pairwise((*n_features, channels_out)):
            if batch_norm:
                self.append(BatchNormND(dim)(c_in, affine=not use_film))
            if use_film:
                self.append(FiLM(c_in, cond_dim))
            self.append(ReLU(True))
            self.append(ConvND(dim)(c_in, c_out, kernel_size=3, padding='same'))

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None) -> torch.Tensor:
        """Apply the basic CNN to the input tensor.

        Parameters
        ----------
        x
            The input tensor. Should be of shape `(batch_size, channels_in, *spatial dimensions)`
            with `spatial dimensions` being of length `dim`.
        cond
            The condition tensor. If None, no FiLM conditioning is applied.

        Returns
        -------
            The output tensor.
        """
        return super().__call__(x, cond=cond)
