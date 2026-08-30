"""DropPath (stochastic depth)."""

import torch
from torch.nn import Module


class DropPath(Module):
    """Drop path or stochastic depth.

    Drops full samples from batch with probability `droprate`.
    Should be used in the main path of a Resblock.

    References
    ----------
    .. [HUANG16] Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. Q. Deep networks with stochastic depth.
       ECCV 2016. https://link.springer.com/chapter/10.1007/978-3-319-46493-0_39
    """

    def __init__(self, droprate: float = 0.0, scale_by_keep: bool = False):
        """Initialize the DropPath module.

        Parameters
        ----------
        droprate
            Drop probability
        scale_by_keep
            If True, the kept samples are scaled by :math:`1/(1-droprate)`
        """
        super().__init__()
        self.droprate = droprate
        self.scale_by_keep = scale_by_keep

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DropPath.

        Parameters
        ----------
        x
            Input tensor

        Returns
        -------
            Tensor with batch samples randomly dropped
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DropPath."""
        if self.droprate == 0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = ((1 - self.droprate) + torch.rand(shape, dtype=x.dtype, device=x.device)).floor_()
        if self.scale_by_keep:
            mask = mask.div_(1 - self.droprate)
        return x * mask
