"""Transform Domain Thresholding using soft thresholding."""

from __future__ import annotations

from typing import overload

import torch

from mrpro.data.IData import IData
from mrpro.operators.functionals import L1NormViewAsReal
from mrpro.operators.LinearOperator import LinearOperator


@overload
def transform_domain_soft_thresholding(
    idata: IData,
    transform_op: LinearOperator,
    threshold: float | torch.Tensor,
) -> IData: ...


@overload
def transform_domain_soft_thresholding(
    idata: torch.Tensor,
    transform_op: LinearOperator,
    threshold: float | torch.Tensor,
) -> torch.Tensor: ...


def transform_domain_soft_thresholding(
    idata: IData | torch.Tensor,
    transform_op: LinearOperator,
    threshold: float | torch.Tensor,
) -> IData | torch.Tensor:
    r"""Apply domain thresholding.

    This algorithm implements denoising by applying :math:x = W^H \mathrm{SoftThreshold}_l (W y), where :math:y is the
    given noisy image, :math:W is a linear operator, and \mathrm{SoftThreshold}_l is a thresholding function with
    parameter :math:l>0.

    This operator is well-known in the signal-processing community under the name of "Domain Transform Thresholding"
    and has various applications and interpretations, see for example [AHMAD2020]_. For instance, when :math:W is
    orthogonal and \mathrm{SoftThreshold}_l:=\mathrm{SoftThreshold}_l, the algorithm gives the closed-form solution of
    the problem :math:min_x \frac{1}{2}||x - y||_2^2 + l ||Wx||_1.

    References
    ----------
    .. [AHMAD2020] Ahmad R, Bouman CA, Buzzard GT, Chan S, Liu S, Reehorst ET, Schniter P (2020) Plug-and-Play Methods
        for Magnetic Resonance Imaging: Using Denoisers for Image Recovery. IEEE Signal Process Mag 37(1):105-116.
        https://doi.org/10.1109/msp.2019.2949470


    Parameters
    ----------
    idata
        input image
    transform_op
        linear operator representing the transformation from image space to transform space, in which the coefficients
        will be thresholded
    threshold
        Threshold for soft thresholding


    Returns
    -------
        the thresholded image.
    """
    img_tensor = idata if isinstance(idata, torch.Tensor) else idata.data

    (coefficients,) = transform_op(img_tensor)
    (thresholded_coefficients,) = L1NormViewAsReal().prox(coefficients, threshold)
    (img_tensor,) = transform_op.H(thresholded_coefficients)

    return img_tensor if isinstance(idata, torch.Tensor) else IData(img_tensor, idata.header)
