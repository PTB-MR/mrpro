"""Basic CNN."""

from collections.abc import Sequence
from itertools import pairwise
from typing import Literal

import torch
from torch.nn import LeakyReLU, ReLU, SiLU

from mrpro.nn.FiLM import FiLM
from mrpro.nn.GroupNorm import GroupNorm
from mrpro.nn.ndmodules import BatchNormND, ConvND
from mrpro.nn.Sequential import Sequential


class BasicCNN(Sequential):
    """Basic CNN.

    A series of convolutions (window 3, stride 1, padding 1), normalization and activation.
    Allows to use FiLM conditioning.
    Order is Conv -> Norm (optional) -> FiLM (optional) -> Activation.

    If you need more flexibility, use `~mrpro.nn.Sequential` directly.
    """

    def __init__(
        self,
        dim: int,
        channels_in: int,
        channels_out: int,
        norm: Literal['batch', 'group', 'instance', 'none', 'layer'] = 'none',
        activation: Literal['relu', 'silu', 'leaky_relu'] = 'relu',
        n_features: Sequence[int] = (64, 64, 64),
        cond_dim: int = 0,
    ):
        """Initialize a basic CNN.

        Parameters
        ----------
        dim
            The number of spatial dimensions of the input tensor.
        channels_in
            The number of input channels.
        channels_out
            The number of output channels.
        norm
            The type of normalization to use. If 'batch', use batch normalization. If 'group', use group normalization,
            if 'instance', use instance normalization, and if `layer`, use layer normalization.
            If 'none', use no normalization.
        n_features
            The number of features in the hidden layers. The length of this sequence determines the number of hidden
            layers. The total number of convolutions is `len(n_features) + 1`.
        cond_dim
            The dimension of the condition tensor. If 0, no FiLM conditioning is applied.
            Otherwise, between convolutions, after normalization, FiLM conditioning is applied.
        """
        super().__init__()
        use_film = cond_dim > 0

        self.append(ConvND(dim)(channels_in, n_features[0], kernel_size=3, padding='same'))

        for c_in, c_out in pairwise((*n_features, channels_out)):
            if norm.lower() == 'batch':
                self.append(BatchNormND(dim)(c_in, affine=not use_film))
            elif norm.lower() == 'group':
                self.append(GroupNorm(c_in, affine=not use_film))
            elif norm.lower() == 'instance':
                self.append(GroupNorm(c_in, groups=c_in, affine=not use_film))  # is instance norm
            elif norm.lower() == 'layer':
                self.append(GroupNorm(c_in, groups=1, affine=not use_film))  # is layer norm
            elif norm.lower() != 'none':
                raise ValueError(f'Invalid normalization type: {norm}')

            if use_film:
                self.append(FiLM(c_in, cond_dim))

            if activation.lower() == 'relu':
                self.append(ReLU(True))
            elif activation.lower() == 'silu':
                self.append(SiLU(inplace=True))
            elif activation.lower() == 'leaky_relu':
                self.append(LeakyReLU(inplace=True))
            else:
                raise ValueError(f'Invalid activation type: {activation}')

            self.append(ConvND(dim)(c_in, c_out, kernel_size=3, padding='same'))

    def __call__(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
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
        return super().__call__(*x, cond=cond)
