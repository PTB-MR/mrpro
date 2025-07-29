"""Gated linear unit activation function."""

import torch
from torch.nn import Linear, Module


class GEGLU(Module):
    r"""Gated linear unit activation function.

    References
    ----------
    ..[GLU] Shazeer, N. (2020). GLU variants improve transformer. https://arxiv.org/abs/2002.05202
    """

    def __init__(self, n_channels_in: int, n_channels_out: int | None = None, features_last: bool = False):
        """Initialize the GEGLU activation function.

        Parameters
        ----------
        n_channels_in
            The number of input features/channels.
        n_channels_out
            The number of output features/channels. If None, the number of
            output features is the same as the number of input features.
        features_last
            If True, the channel dimension is the last dimension, else in the second dimension.
        """
        super().__init__()
        out_channels_ = n_channels_in if n_channels_out is None else n_channels_out
        self.proj = Linear(n_channels_in, out_channels_ * 2)  # gate and output stacked
        self.features_last = features_last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the GEGLU activation."""
        if not self.features_last:
            x = x.moveaxis(1, -1)
        h, gate = self.proj(x).chunk(2, dim=-1)
        gate = torch.nn.functional.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)
        out = h * gate
        if not self.features_last:
            out = out.moveaxis(-1, 1)
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the GEGLU activation.

        Parameters
        ----------
        x
            Input tensor

        Returns
        -------
            Activated tensor
        """
        return super().__call__(x)
