"""Attention gate from Attention UNet."""

import torch
from torch.nn import Module, ReLU, Sequential, Sigmoid

from mr2.nn.ndmodules import convND


class AttentionGate(Module):
    """Attention gate from Attention UNet.

    The attention mechanism from the attention UNet [OKT18]_.

    References
    ----------
    ..[OKT18] Oktay, Ozan, et al. "Attention U-net: Learning where to look for the pancreas." MIDL (2018).
      https://arxiv.org/abs/1804.03999
    """

    def __init__(
        self, n_dim: int, channels_gate: int, channels_in: int, channels_hidden: int, concatenate: bool = False
    ):
        """Initialize the attention gate.

        Parameters
        ----------
        n_dim
            The dimension, i.e. 1, 2 or 3.
        channels_gate
            The number of channels in the gate tensor.
        channels_in
            The number of channels in the input tensor.
        channels_hidden
            The number of internal, hidden channels.
        concatenate
            Whether to concatenate the gated signal with the gate signal in the channel dimension (1)
        """
        super().__init__()
        self.project_gate = convND(n_dim)(channels_gate, channels_hidden, kernel_size=1)
        self.project_x = convND(n_dim)(channels_in, channels_hidden, kernel_size=1)
        self.psi = Sequential(
            ReLU(),
            convND(n_dim)(channels_hidden, 1, kernel_size=1),
            Sigmoid(),
        )
        self.concatenate = concatenate

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
        projected_gate = self.project_gate(gate)
        projected_x = self.project_x(x)
        projected_gate = torch.nn.functional.interpolate(projected_gate, size=x.shape[2:], mode='nearest')
        alpha = self.psi(projected_gate + projected_x)
        x = x * alpha
        if self.concatenate:
            gate = torch.nn.functional.interpolate(gate, size=x.shape[2:], mode='nearest')
            x = torch.cat([x, gate], dim=1)
        return x
