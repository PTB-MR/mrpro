"""GroupNorm with 32-bit precision."""

import torch


class GroupNorm32(torch.nn.GroupNorm):
    """A 32-bit GroupNorm.

    Casts to float32 before calling the parent class to avoid instabilities in mixed precision training.
    """

    def __init__(self, channels: int, groups: int | None = None):
        """Initialize GroupNorm32.

        Parameters
        ----------
        channels
            The number of channels in the input tensor.
        groups
            The number of groups to use. If None, the number of groups is determined automatically as
            a power of 2 that is less than or equal to 32 and leaves at least 4 channels per group.
        """
        if groups is None:
            groups_ = channels & -channels
            while (groups_ >= channels // 4) or groups_ > 32:
                groups_ //= 2
        else:
            groups_ = groups
        super().__init__(groups_, channels)

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
        """Apply GroupNorm32."""
        return super().forward(x.float()).type(x.dtype)
