"""Simple Convolutional Neural Network."""

from collections.abc import Sequence
from itertools import pairwise

from torch.nn import ReLU

from mrpro.nn.FiLM import FiLM
from mrpro.nn.GroupNorm import GroupNorm
from mrpro.nn.NDModules import ConvND
from mrpro.nn.Residual import Residual
from mrpro.nn.Sequential import Sequential


class CNN(Sequential):
    """A simple CNN network."""

    def __init__(
        self,
        dim: int,
        channels_in: int,
        channels_out: int,
        features: Sequence[int],
        norm: bool = True,
        residual: bool = True,
        cond_dim: int = 0,
    ):
        """Initialize the CNN.

        Parameters
        ----------
        dim
            The number of spatial dimensions.
        channels_in
            The number of input channels.
        channels_out
            The number of output channels.
        features
            The number of features in each layer. The length of the list is the number of hidden layers.
        norm
            Whether to use layer normalization.
        residual
            Whether to use residual connections.
        cond_dim
            The dimension of the conditioning tensor. If 0, no FiLM is used.
        """
        super().__init__()
        channels = [channels_in, *features]
        for i, (channels_current, channels_next) in enumerate(pairwise(channels)):
            block = Sequential(ConvND(dim)(channels_current, channels_next, 3, padding=1), ReLU(True))
            if norm:
                block.append(GroupNorm(1))
            if cond_dim > 0 and i % 2 == 0:
                block.append(FiLM(channels_next, cond_dim))
            if residual:
                self.append(Residual(block))
            else:
                self.append(block)

        self.append(ConvND(dim)(channels_next, channels_out, 3, padding=1))
