"""Attention gate from Attention UNet."""

import torch
from torch.nn import Module, ReLU, Sequential, Sigmoid

from mrpro.nn.NDModules import ConvND


class AttentionGate(Module):
    """Attention gate from Attention UNet.

    The attention mechanism from the attention UNet [OKT18]_.

    References
    ----------
    ..[OKT18] Oktay, Ozan, et al. "Attention u-net: Learning where to look for the pancreas." MIDL (2018).
      https://arxiv.org/abs/1804.03999
    """

    def __init__(self, dim: int, channels_gate: int, channels_in: int, channels_hidden: int):
        """Initialize the attention gate.

        Parameters
        ----------
        dim
            The dimension, i.e. 1, 2 or 3.
        channels_gate
            The number of channels in the gate tensor.
        channels_in
            The number of channels in the input tensor.
        channels_hidden
            The number of internal, hidden channels.
        """
        super().__init__()
        self.project_gate = ConvND(dim)(channels_gate, channels_hidden, kernel_size=1)
        self.project_x = ConvND(dim)(channels_in, channels_hidden, kernel_size=1)
        self.psi = Sequential(
            ReLU(),
            ConvND(dim)(channels_hidden, 1, kernel_size=1),
            Sigmoid(),
        )

    def __call__(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Apply the attention gate.

        Parameters
        ----------
        x
            The input tensor.
        gate
            The gate tensor.

        Returns
        -------
            The output tensor.
        """
        return super().__call__(x, gate)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Apply the attention gate."""
        gate = self.project_gate(gate)
        x = self.project_x(x)
        alpha = self.psi(gate + x)
        return x * alpha
