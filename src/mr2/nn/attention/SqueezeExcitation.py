"""Squeeze-and-Excitation block."""

import torch
from torch.nn import Module, ReLU, Sigmoid

from mr2.nn.ndmodules import adaptiveAvgPoolND, convND
from mr2.nn.Sequential import Sequential


class SqueezeExcitation(Module):
    """Squeeze-and-Excitation block.

    Sequeeze-and-Excitation block from [SE]_.

    References
    ----------
    ..[SE] Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." CVPR 2018, https://arxiv.org/abs/1709.01507
    """

    def __init__(self, n_dim: int, n_channels_input: int, n_channels_squeeze: int) -> None:
        """Initialize SqueezeExcitation.

        Parameters
        ----------
        n_dim
            The dimension of the input tensor.
        n_channels_input
            The number of channels in the input tensor.
        n_channels_squeeze
            The number of channels in the squeeze tensor.
        """
        super().__init__()
        self.scale = Sequential(
            adaptiveAvgPoolND(n_dim)(1),
            convND(n_dim)(n_channels_input, n_channels_squeeze, kernel_size=1),
            ReLU(),
            convND(n_dim)(n_channels_squeeze, n_channels_input, kernel_size=1),
            Sigmoid(),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SqueezeExcitation.

        Parameters
        ----------
        x
            The input tensor.

        Returns
        -------
            The output tensor.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SqueezeExcitation."""
        return x * self.scale(x)
