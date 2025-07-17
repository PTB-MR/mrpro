"""RMSNorm over the channel dimension."""

import torch
from torch.nn import Module, Parameter


class RMSNorm(Module):
    """RMSNorm over the channel dimension.

    As used in the DCAE [DCAE]_.

    References
    ----------
    .. [DCAE] Chen, J., Cai, H., Chen, J., Xie, E., Yang, S., Tang, H., ... & Han, S. Deep compression autoencoder
       for efficient high-resolution diffusion models. ICLR 2025. https://arxiv.org/abs/2410.10733
    """

    def __init__(
        self,
        n_channels: int | None = None,
        eps: float = 1e-8,
        features_last: bool = False,
    ):
        """Initialize RMSNorm.

        Includes a learnable weight and bias if n_channels is provided.

        Parameters
        ----------
        n_channels
            Number of channels. If `None`, no learnable weight and bias are included.
        eps
            Epsilon value to avoid division by zero.
        features_last
            If True, the channel dimension is the last dimension.
        """
        super().__init__()
        if n_channels is not None:
            self.weight: Parameter | None = Parameter(torch.zeros(n_channels))
            self.bias: Parameter | None = Parameter(torch.zeros(n_channels))
        else:
            self.weight = None
            self.bias = None
        self.eps = eps
        self.channel_dim = -1 if features_last else 1

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm over the channel dimension.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            Normalized tensor.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm over the channel dimension."""
        x32 = x.to(torch.float32)  # normalization in float32 to stabilize mixed precision training
        mean_square = x32.square().mean(dim=self.channel_dim, keepdim=True)
        scale = (mean_square + self.eps).rsqrt()
        x32 = x32 * scale
        if self.weight is not None and self.bias is not None:
            shape = [1] * x.ndim
            shape[self.channel_dim] = -1
            weight = (self.weight.to(x32.dtype) + 1).view(shape)
            bias = self.bias.view(shape)
            x32 = x32 * weight + bias
        return x32.to(x.dtype)
