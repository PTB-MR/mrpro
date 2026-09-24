"""Wavelet Denoising using soft thresholding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import torch

from mrpro.data.IData import IData
from mrpro.operators.functionals import L1NormViewAsReal
from mrpro.operators.WaveletOp import WaveletOp, WaveletType
from mrpro.utils import normalize_index


@overload
def wavelet_denoising(
    idata: IData,
    regularization_dim: Sequence[int],
    regularization_weight: float | torch.Tensor,
    wavelet_name: WaveletType = 'db4',
    level: int | None = None,
) -> IData: ...


@overload
def wavelet_denoising(
    idata: torch.Tensor,
    regularization_dim: Sequence[int],
    regularization_weight: float | torch.Tensor,
    wavelet_name: WaveletType = 'db4',
    level: int | None = None,
) -> torch.Tensor: ...


def wavelet_denoising(
    idata: IData | torch.Tensor,
    regularization_dim: Sequence[int],
    regularization_weight: float | torch.Tensor,
    wavelet_name: WaveletType = 'db4',
    level: int | None = None,
) -> IData | torch.Tensor:
    r"""Apply wavelet denoising.

    This algorithm solves the problem :math:`min_x \frac{1}{2}||x - y||_2^2 + l ||Wx||_1`
    in closed form as :math:`x = W^H \mathrm{SoftThreshold}_l (Wy)`. :math:`y` is the given noisy image,
    :math:`l` is the strength of the regularization, :math:`W` is the wavelet operator and the soft
    thresholding is the proximal map of the :math:`L_1` norm.

    .. note::
        The closed-form solution is exact for an orthonormal wavelet transform. Due to the padding at the
        image boundaries, `~mrpro.operators.WaveletOp` returns more coefficients than the image has pixels
        for most combinations of wavelet and image size, and the result is then a close approximation.

    Parameters
    ----------
    idata
        input image
    regularization_dim
        Dimensions along which the wavelet transform is applied.
    regularization_weight
        Strength of the regularization (:math:`l`), i.e. the threshold. Can also be a tensor which is
        broadcastable to the wavelet coefficients, e.g. to use a different threshold per wavelet scale.
    wavelet_name
        Name of the wavelet, see `~mrpro.operators.WaveletOp`.
    level
        Number of wavelet levels. If `None`, maximum number of levels is used.

    Returns
    -------
        the denoised image.
    """
    img_tensor = idata if isinstance(idata, torch.Tensor) else idata.data

    dim = tuple(normalize_index(img_tensor.ndim, idx) - img_tensor.ndim for idx in regularization_dim)
    if len(dim) != len(set(dim)):
        raise ValueError('Repeated values are not allowed in regularization_dim')

    wavelet_op = WaveletOp(
        domain_shape=tuple(img_tensor.shape[d] for d in dim),
        dim=dim,  # type: ignore[arg-type]
        wavelet_name=wavelet_name,
        level=level,
    )

    (coefficients,) = wavelet_op(img_tensor)
    (thresholded_coefficients,) = L1NormViewAsReal().prox(coefficients, regularization_weight)
    (img_tensor,) = wavelet_op.H(thresholded_coefficients)

    return img_tensor if isinstance(idata, torch.Tensor) else IData(img_tensor, idata.header)
