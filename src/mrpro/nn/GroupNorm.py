"""GroupNorm with 32-bit precision."""

import torch


class GroupNorm(torch.nn.GroupNorm):
    """A 32-bit GroupNorm with (optional) automatic group size selection.

    Casts to float32 before calling the parent class to avoid instabilities in mixed precision training.

    If `n_groups` is not provided, the number of groups is selected automatically as follows:

    - start from `1` group,
    - try powers of two (`2, 4, 8, ...`),
    - keep the largest candidate that divides `n_channels`,
    - enforce at most `32` groups and at least `4` channels per group.

    This yields a stable default that stays close to common GroupNorm choices while
    adapting to small channel counts.
    """

    features_last: bool

    def __init__(self, n_channels: int, n_groups: int | None = None, affine: bool = False, features_last: bool = False):
        """Initialize GroupNorm.

        Parameters
        ----------
        n_channels
            The number of channels in the input tensor.
        n_groups
            The number of groups to use. If None, the number of groups is determined automatically as
            the largest power of 2 that divides `n_channels`, is less than or equal to 32,
            and leaves at least 4 channels per group.
        affine
            Whether to use learnable affine parameters.
        features_last
            Whether the features are last in the input tensor, as common in transformer models,
            or in the second dimension, as common in CNNs.
        """
        if n_groups is None:
            groups_, candidate = 1, 2
            while (candidate <= min(32, n_channels // 4)) and (n_channels % candidate == 0):
                groups_, candidate = candidate, groups_ * 2
        else:
            groups_ = n_groups
        self.features_last: bool = features_last
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
        if self.features_last:
            x = x.moveaxis(-1, 1)
        result = super().forward(x.float()).type(x.dtype)
        if self.features_last:
            result = result.moveaxis(1, -1)
        return result
