"""Structural Similarity Index (SSIM) functional."""

from typing import cast

import torch

from mrpro.operators.Functional import ElementaryFunctional
from mrpro.utils.sliding_window import sliding_window


def ssim3d(
    target: torch.Tensor,
    prediction: torch.Tensor,
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
        Ground truth tensor, shape `(... z, y, x)` or broadcastable with prediction.
    prediction
        Predicted tensor, same shape as target
    mask
        Mask tensor, same shape as target.
        Only windows that are fully inside the mask are considered.
        If None, all elements are considered.
        If float, windows will be weighted by the average value of the mask in the window.
    k1
        Constant for SSIM computation. Commonly 0.01.
    k2
        Constant for SSIM computation. Commonly 0.03.
    window_size
        Size of the rectangular sliding window.
        If any of the last 3 dimensions of the target is of size 1, the window size in this dimension will
        also be set to 1.
    data_range
        Value range if the data. If `None`, max-to-min of the target will be used.
    reduction
        If True, return the mean SSIM over all pixels. If False, return the SSIM map.
        The map will be of shape `(... z - window_size,  y - window_size, x - window_size)`

    Returns
    -------
        Scalar mean SSIM or SSIM map.
    """
    if target.ndim < 3:
        raise ValueError('Input must be at least 3D (z, y, x)')

    if mask is not None:
        if (mask < 0).any():
            raise ValueError('Mask contains negative values')
        target, prediction, mask = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.broadcast_tensors(target, prediction, mask)
        )
        mask = sliding_window(mask, window_shape=window_size, dim=(-3, -2, -1)).movedim((0, 1, 2), (-6, -5, -4))
        bool_mask = mask.all(dim=(-3, -2, -1))
        if bool_mask.numel() == 0:
            raise ValueError('Mask does not cover any pixels')
        mask = mask[bool_mask]
        mask = mask / mask.sum()

    else:
        bool_mask = None
        target, prediction = cast(tuple[torch.Tensor, torch.Tensor], torch.broadcast_tensors(target, prediction))

    window = tuple(window_size if s > 1 else 1 for s in target.shape[-3:])  # To support 1D and 2D uses
    target_window = sliding_window(target, window_shape=window, dim=(-3, -2, -1)).movedim((0, 1, 2), (-6, -5, -4))

    if data_range is None and bool_mask is not None:
        data_range_ = target_window[bool_mask].amax() - target_window[bool_mask].amin()
    if data_range is None:
        data_range_ = target_window.amax() - target_window.amin()
    else:
        data_range_ = data_range

    x_window = sliding_window(prediction, window_shape=window, dim=(-3, -2, -1)).movedim((0, 1, 2), (-6, -5, -4))

    mean_tgt = target_window.mean(dim=(-3, -2, -1))
    mean_img = x_window.mean(dim=(-3, -2, -1))

    mean_tgt_tgt = (target_window**2).mean(dim=(-3, -2, -1))
    mean_img_img = (x_window**2).mean(dim=(-3, -2, -1))
    mean_tgt_img = (target_window * x_window).mean(dim=(-3, -2, -1))

    n = x_window.shape[-3:].numel()
    cov_norm = n / (n - 1)
    cov_tgt = cov_norm * (mean_tgt_tgt - mean_tgt * mean_tgt)
    cov_img = cov_norm * (mean_img_img - mean_img * mean_img)
    cov_tgt_img = cov_norm * (mean_tgt_img - mean_tgt * mean_img)

    c1 = (k1 * data_range_) ** 2
    c2 = (k2 * data_range_) ** 2
    a1 = 2 * mean_tgt * mean_img + c1
    a2 = 2 * cov_tgt_img + c2
    b1 = mean_tgt**2 + mean_img**2 + c1
    b2 = cov_tgt + cov_img + c2

    ssim_map = (a1 * a2) / (b1 * b2)

    if reduction and bool_mask is not None and mask is not None:
        return (ssim_map[bool_mask] * mask).sum()
    elif reduction and bool_mask is None:
        return ssim_map.mean()
    elif not reduction and bool_mask is not None:
        return torch.where(bool_mask, ssim_map, ssim_map[bool_mask].mean())
    else:  # reduction and mask is None
        return ssim_map


class SSIM(ElementaryFunctional):
    """(masked) SSIM functional."""

    def __init__(
        self,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        data_range: torch.Tensor | None = None,
        window_size: int = 7,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> None:
        """Initialize Volume SSIM.

        The Structural Similarity Index Measure [SSIM]_ is used to measure the similarity between two images.
        It considers luminance, contrast and structure differences between the images.
        SSIM values range from -1 to 1, where 1 indicates perfect structural similarity.

        Calculates the SSIM using a rectangular sliding window. If a mask is provided, only the windows
        that are fully inside the mask are considered.

        The SSIM is calculated for a volume, i.e, 3D patches of the last three dimensions of the input are used.
        To apply it to purely 2D data, add a singleton dimension.

        References
        ----------
        .. [SSIM] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
               Image quality assessment: from error visibility to structural similarity.
               IEEE TMI, 13(4), 600-612. https://doi.org/10.1109/TIP.2003.819861



        Parameters
        ----------
        target
            Target volume. At least 3d.
        weight
            Either a boolean mask. Only windows with all values `True` will be considered.
            `None` means all windows are used.
            Or a weight tensor of the same shape as the target. Each window will be weighted by
            the average value of the weight in the window.
        data_range
            Value range if the data. If None, the max-to-min of the target will be used.
        window_size
            Size of the windows used in SSIM. Usually 7 or 11.
            Will be cropped to
        k1
            Constant. Usually 0.01
        k2
            Constant. Usually 0.03
        """
        weight_ = 1 if weight is None else weight
        super().__init__(target, weight_)
        self.k1 = k1
        self.k2 = k2
        self.window_size = window_size
        self.data_range = data_range

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Calculate SSIM of an input."""
        if x.is_complex():
            return ((self.forward(x.real)[0] + self.forward(x.imag)[0]) / 2,)
        return (
            ssim3d(
                self.target,
                x,
                mask=self.weight,
                k1=self.k1,
                k2=self.k2,
                window_size=self.window_size,
                data_range=self.data_range,
            ),
        )
