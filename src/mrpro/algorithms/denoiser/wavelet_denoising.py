"""Wavelet Denoising using soft thresholding."""

from __future__ import annotations

from typing import overload

import pywt
import torch

from mrpro.algorithms.denoiser.transform_domain_soft_thresholding import transform_domain_soft_thresholding
from mrpro.data.IData import IData
from mrpro.operators.WaveletOp import WaveletOp, WaveletType
from mrpro.utils import normalize_index


@overload
def wavelet_denoising(
    idata: IData,
    regularization_dim: tuple[int] | tuple[int, int] | tuple[int, int, int],
    regularization_weight: float | torch.Tensor,
    wavelet_name: WaveletType = 'db4',
    level: int | None = None,
) -> IData: ...


@overload
def wavelet_denoising(
    idata: torch.Tensor,
    regularization_dim: tuple[int] | tuple[int, int] | tuple[int, int, int],
    regularization_weight: float | torch.Tensor,
    wavelet_name: WaveletType = 'db4',
    level: int | None = None,
) -> torch.Tensor: ...


def wavelet_denoising(
    idata: IData | torch.Tensor,
    regularization_dim: tuple[int] | tuple[int, int] | tuple[int, int, int],
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
        Dimensions along which the wavelet transform is applied (1, 2 or 3 dimensions).
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
    if len({normalize_index(idata.ndim, idx) for idx in regularization_dim}) != len(regularization_dim):
        raise ValueError('Repeated values are not allowed in regularization_dim')

    # Continuous wavelets (gaus, mexh, morl, cgau, shan, fbsp, cmor)
    # are not in the discrete list and have no orthogonal filter bank.
    if wavelet_name not in pywt.wavelist(kind='discrete'):
        raise NotImplementedError(f"Wavelet '{wavelet_name}' is a continuous wavelet and is not orthonormal.")
    if not pywt.Wavelet(wavelet_name).orthogonal:
        raise NotImplementedError(f"Wavelet '{wavelet_name}' is not orthonormal.")

    return transform_domain_soft_thresholding(
        idata=idata,
        transform_op=WaveletOp(
            domain_shape=tuple(idata.shape[d] for d in regularization_dim),
            dim=regularization_dim,
            wavelet_name=wavelet_name,
            level=level,
        ),
        threshold=regularization_weight,
    )
