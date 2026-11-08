"""Wavelet Denoising using soft thresholding."""

from __future__ import annotations

from collections.abc import Callable
from typing import overload

import torch

from mrpro.data.IData import IData
from mrpro.operators.LinearOperator import LinearOperator


@overload
def domain_thresholding(
    idata: IData,
    domain_op: LinearOperator,
    threshold_function: Callable[[torch.Tensor, float], torch.Tensor],
    threshold: float,
) -> IData: ...


@overload
def domain_thresholding(
    idata: torch.Tensor,
    domain_op: LinearOperator,
    threshold_function: Callable[[torch.Tensor, float], torch.Tensor],
    threshold: float,
) -> torch.Tensor: ...


def domain_thresholding(
    idata: IData | torch.Tensor,
    domain_op: LinearOperator,
    threshold_function: Callable[[torch.Tensor, float], torch.Tensor],
    threshold: float,
) -> IData | torch.Tensor:
    r"""Apply domain thresholding.

    This algorithm solves the problem :math:`min_x \frac{1}{2}||x - y||_2^2 + l ||Wx||_1`
    in closed form as :math:`x = W^H \mathrm{SoftThreshold}_l (Wy)`. :math:`y` is the given noisy image,
    :math:`l` is the strength of the regularization, :math:`W` is the wavelet operator and the soft
    thresholding is the proximal map of the :math:`L_1` norm.

    Parameters
    ----------
    idata
        input image
    domain_op
        linear operator representing the domain transformation
    threshold_function
        function to apply thresholding
    threshold
        Threshold for thresholding function


    Returns
    -------
        the denoised image.
    """
    img_tensor = idata if isinstance(idata, torch.Tensor) else idata.data

    (coefficients,) = domain_op(img_tensor)
    (thresholded_coefficients,) = threshold_function(coefficients, threshold)
    (img_tensor,) = domain_op.H(thresholded_coefficients)

    return img_tensor if isinstance(idata, torch.Tensor) else IData(img_tensor, idata.header)
