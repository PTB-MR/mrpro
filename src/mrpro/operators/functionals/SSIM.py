"""Structural Similarity Index (SSIM) functional."""

from typing import cast

import torch

from mrpro.operators.Functional import Functional
from mrpro.utils.sliding_window import sliding_window


def ssim3d(
    target: torch.Tensor,
    x: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
    window_size: int = 7,
    data_range: torch.Tensor | None = None,
    reduction: bool = True,
) -> torch.Tensor:
    """Compute SSIM between two 3D volumes.

    Parameters
    ----------
    target
        Ground truth tensor, shape `(... z, y, x)`
    x
        Predicted tensor, same shape as gt
    mask
        Mask tensor, same shape as target.
        Only windows that are fully inside the mask are considered.
        If None, all elements are considered.
    k1
        Constant for SSIM computation. Commonly 0.01.
    k2
        Constant for SSIM computation. Commonly 0.03.
    window_size
        Size of the rectangular sliding window.
    data_range
        Value range if the data. If `None`, peak-to-peak of the target will be used.
    reduction
        If True, return the mean SSIM over all pixels. If False, return the SSIM map.
        The map will be of shape `(... z-window_size, y-window_size, x-window_size)`

    Returns
    -------
        Scalar mean SSIM or SSIM map.
    """
    if target.ndim < 3:
        raise ValueError('Input must be at least 3D (z, y, x)')

    if mask is not None:
        target, x, mask = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.broadcast_tensors(target, x, mask)
        )
        target = target * mask
        x = x * mask
        mask = sliding_window(mask, window_shape=window_size, dim=(-3, -2, -1)).movedim((0, 1, 2), (-6, -5, -4))
        mask = mask.all(dim=(-3, -2, -1))
    else:
        target, x = cast(tuple[torch.Tensor, torch.Tensor], torch.broadcast_tensors(target, x))

    if data_range is None:
        data_range_ = torch.amax(target, dim=(-3, -2, -1), keepdim=True) - torch.amin(
            target, dim=(-3, -2, -1), keepdim=True
        )
        data_range_ = data_range_.clamp_min(1e-6)
    else:
        data_range_ = data_range

    target_window = sliding_window(target, window_shape=window_size, dim=(-3, -2, -1)).movedim((0, 1, 2), (-6, -5, -4))
    x_window = sliding_window(x, window_shape=window_size, dim=(-3, -2, -1)).movedim((0, 1, 2), (-6, -5, -4))

    mean_t = target_window.mean(dim=(-3, -2, -1))
    mean_x = x_window.mean(dim=(-3, -2, -1))

    mean_tt = (target_window**2).mean(dim=(-3, -2, -1))
    mean_xx = (x_window**2).mean(dim=(-3, -2, -1))
    mean_tx = (target_window * x_window).mean(dim=(-3, -2, -1))

    n = x_window.shape[-3:].numel()
    cov_norm = n / (n - 1)
    cov_t = cov_norm * (mean_tt - mean_t * mean_t)
    cov_x = cov_norm * (mean_xx - mean_x * mean_x)
    cov_tx = cov_norm * (mean_tx - mean_t * mean_x)

    c1 = (k1 * data_range_) ** 2
    c2 = (k2 * data_range_) ** 2
    a1 = 2 * mean_t * mean_x + c1
    a2 = 2 * cov_tx + c2
    b1 = mean_t**2 + mean_x**2 + c1
    b2 = cov_t + cov_x + c2

    ssim_map = (a1 * a2) / (b1 * b2)

    if reduction and mask is not None:
        return ssim_map[mask].mean()
    elif reduction and mask is None:
        return ssim_map.mean()
    elif not reduction and mask is not None:
        return torch.where(mask, ssim_map, ssim_map[mask].mean())
    else:  # reduction and mask is None
        return ssim_map


class SSIM(Functional):
    """(masked) SSIM functional."""

    def __init__(
        self,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        data_range: torch.Tensor | None = None,
        window_size: int = 7,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> None:
        """Initialize SSIM.

        The Structural Similarity Index Measure [SSIM]_ is used to measure the similarity between two images.
        It considers luminance, contrast and structure differences between the images.
        SSIM values range from -1 to 1, where 1 indicates perfect structural similarity.

        Calculates the SSIM using a rectangular sliding window. If a mask is provided, only the windows
        that are fully inside the mask are considered.

        References
        ----------
        .. [SSIM] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
               Image quality assessment: from error visibility to structural similarity.
               IEEE TMI, 13(4), 600-612. https://doi.org/10.1109/TIP.2003.819861



        Parameters
        ----------
        target
            Target volume. At least 3d.
        mask
            Boolean mask. Only windows with all values `True` will be considered.
            `None` means all windows are used.
        data_range
            Value range if the data. If None, the peak-to-peak of the target will be used.
        window_size
            Size of the windows used in SSIM. Usually 7 or 11.
        k1
            Constant. Usually 0.01
        k2
            Constant. Usually 0.03
        """
        super().__init__()
        self.target = target
        self.mask = mask
        self.k1 = k1
        self.k2 = k2
        self.window_size = window_size
        self.data_range = data_range
        if mask is not None and mask.dtype != torch.bool:
            raise ValueError('Mask should be a boolean tensor')

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Calculate SSIM of an input."""
        return (
            ssim3d(
                self.target,
                x,
                mask=self.mask,
                k1=self.k1,
                k2=self.k2,
                window_size=self.window_size,
                data_range=self.data_range,
            ),
        )
