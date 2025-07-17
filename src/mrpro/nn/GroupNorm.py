"""GroupNorm with 32-bit precision."""

import torch


class GroupNorm(torch.nn.GroupNorm):
    """A 32-bit GroupNorm with (optional) automatic group size selection.

    Casts to float32 before calling the parent class to avoid instabilities in mixed precision training.
    """

    def __init__(self, n_channels: int, n_groups: int | None = None, affine: bool = False):
        """Initialize GroupNorm.

        Parameters
        ----------
        n_channels
            The number of channels in the input tensor.
        n_groups
            The number of groups to use. If None, the number of groups is determined automatically as
            a power of 2 that is less than or equal to 32 and leaves at least 4 channels per group.
        affine
            Whether to use learnable affine parameters.
        """
        if n_groups is None:
            groups_, candidate = 1, 2
            while (candidate <= min(32, n_channels // 4)) and (n_channels % candidate == 0):
                groups_, candidate = candidate, groups_ * 2
        else:
            groups_ = n_groups
        super().__init__(groups_, n_channels, affine=affine)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GroupNorm32.

        Parameters
        ----------
        x
            The input tensor.

        Returns
        -------
            The output tensor.
        """
        return super().__call__(x.float()).type(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GroupNorm."""
        return super().forward(x.float()).type(x.dtype)
