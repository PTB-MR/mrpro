"""Squeeze-and-Excitation block."""

from torch.nn import Module, ReLU, Sigmoid

from mrpro.nn.NDModules import AdaptiveAvgPoolND, ConvND
from mrpro.nn.Sequential import Sequential
import torch


class SqueezeExcitation(Module):
    """Squeeze-and-Excitation block.

    Sequeeze-and-Excitation block from [SE]_.

    References
    ----------
    ..[SE] Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." CVPR 2018, https://arxiv.org/abs/1709.01507
    """

    def __init__(self, dim: int, input_channels: int, squeeze_channels: int) -> None:
        """Initialize SqueezeExcitation.

        Parameters
        ----------
        dim
            The dimension of the input tensor.
        input_channels
            The number of channels in the input tensor.
        squeeze_channels
            The number of channels in the squeeze tensor.
        """
        super().__init__()
        self.scale = Sequential(
            AdaptiveAvgPoolND(dim)(1),
            ConvND(dim)(input_channels, squeeze_channels, kernel_size=1),
            ReLU(),
            ConvND(dim)(squeeze_channels, input_channels, kernel_size=1),
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
