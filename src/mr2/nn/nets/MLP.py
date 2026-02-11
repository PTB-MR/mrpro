"""Multi-layer perceptron."""

from collections.abc import Sequence
from itertools import pairwise
from typing import Literal

import torch
from torch.nn import GELU, LeakyReLU, Linear, ReLU, SiLU

from mr2.nn.FiLM import FiLM
from mr2.nn.LayerNorm import LayerNorm
from mr2.nn.Sequential import Sequential


class MLP(Sequential):
    """Multi-layer perceptron.

    A series of linear layers, normalization and activation.
    Allows FiLM conditioning.
    Order is Linear -> Norm (optional) -> FiLM (optional) -> Activation.

    If you need more flexibility, use `~mr2.nn.Sequential` directly.
    """

    features_last: bool

    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        norm: Literal['layer', 'none'] = 'none',
        activation: Literal['gelu', 'relu', 'silu', 'leaky_relu'] = 'gelu',
        n_features: Sequence[int] = (256, 256),
        cond_dim: int = 0,
        features_last: bool = True,
    ):
        """Initialize a MLP.

        Parameters
        ----------
        n_channels_in
            The number of input channels.
        n_channels_out
            The number of output channels.
        norm
            The type of normalization to use. If `layer`, use layer normalization.
            If `none`, use no normalization.
        activation
            The type of activation to use. If `gelu`, use GELU.
            If `relu`, use ReLU. If `silu`, use SiLU. If `leaky_relu`, use LeakyReLU.
        n_features
            The number of features in the hidden layers. The length of this sequence determines the number of hidden
            layers. The total number of linear layers is `len(n_features) + 1`.
        cond_dim
            The dimension of the condition tensor. If 0, no FiLM conditioning is applied.
            Otherwise, between linear layers, after normalization, FiLM conditioning is applied.
        features_last
            Whether the features are in the last dimension, as common in transformer models,
            or in the second dimension, as common in image models.
        """
        super().__init__()
        use_film = cond_dim > 0
        self.features_last = features_last

        if len(n_features) == 0:
            self.append(Linear(n_channels_in, n_channels_out))
            return

        self.append(Linear(n_channels_in, n_features[0]))

        for c_in, c_out in pairwise((*n_features, n_channels_out)):
            if norm.lower() == 'layer':
                self.append(LayerNorm(c_in, features_last=True))
            elif norm.lower() != 'none':
                raise ValueError(f'Invalid normalization type: {norm}')

            if use_film:
                self.append(FiLM(c_in, cond_dim, features_last=True))

            if activation.lower() == 'gelu':
                self.append(GELU(approximate='tanh'))
            elif activation.lower() == 'relu':
                self.append(ReLU())
            elif activation.lower() == 'silu':
                self.append(SiLU())
            elif activation.lower() == 'leaky_relu':
                self.append(LeakyReLU())
            else:
                raise ValueError(f'Invalid activation type: {activation}')

            self.append(Linear(c_in, c_out))

    def __call__(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        """Apply the MLP to the input tensor.

        Parameters
        ----------
        x
            The input tensor.
        cond
            The condition tensor. If None, no FiLM conditioning is applied.

        Returns
        -------
            The output tensor.
        """
        return super().__call__(x, cond=cond)

    def forward(self, *x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the MLP to the input tensor."""
        if len(x) != 1:
            raise ValueError(f'Mlp expects exactly one input tensor, got {len(x)}')
        tensor = x[0]
        if not self.features_last:
            tensor = tensor.moveaxis(1, -1)
        out = super().forward(tensor, cond=cond)
        if not self.features_last:
            out = out.moveaxis(-1, 1)
        return out
