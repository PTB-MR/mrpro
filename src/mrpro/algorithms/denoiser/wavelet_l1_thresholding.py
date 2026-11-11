"""Wavelet L1 Thresholding for denoising."""

from __future__ import annotations

from typing import overload

import torch

from mrpro.algorithms.denoiser.transform_domain_soft_thresholding import transform_domain_soft_thresholding
from mrpro.data.IData import IData
from mrpro.operators.WaveletOp import WaveletOp, WaveletType
from mrpro.utils import normalize_index


@overload
def wavelet_l1_thresholding(
    idata: IData,
    regularization_dim: tuple[int] | tuple[int, int] | tuple[int, int, int],
    regularization_weight: float | torch.Tensor,
    wavelet_name: WaveletType = 'db4',
    level: int | None = None,
) -> IData: ...


@overload
def wavelet_l1_thresholding(
    idata: torch.Tensor,
    regularization_dim: tuple[int] | tuple[int, int] | tuple[int, int, int],
    regularization_weight: float | torch.Tensor,
    wavelet_name: WaveletType = 'db4',
    level: int | None = None,
) -> torch.Tensor: ...


def wavelet_l1_thresholding(
    idata: IData | torch.Tensor,
    regularization_dim: tuple[int] | tuple[int, int] | tuple[int, int, int],
    regularization_weight: float | torch.Tensor,
    wavelet_name: WaveletType = 'db4',
    level: int | None = None,
) -> IData | torch.Tensor:
    r"""Apply wavelet L1 thresholding.

    This algorithm carries out :math:`x = W^H \mathrm{SoftThreshold}_l (Wy)`. :math:`y` is the given noisy image,
    :math:`l` is the strength of the regularization and :math:`W` is the wavelet operator.

    If :math:`W` is orthogonal than the algorithm gives the closed-form solution of the problem
    :math:`min_x \frac{1}{2}||x - y||_2^2 + l ||Wx||_1`.

    .. note::
        For padding at the image boundaries, `~mrpro.operators.WaveletOp` returns more coefficients than the image has
        pixels for most combinations of wavelet and image size. In this case the above is also only a close
        approximation and not the true solution even for orthogonal wavelets.

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
        the thresholded image.
    """
    if len({normalize_index(idata.ndim, idx) for idx in regularization_dim}) != len(regularization_dim):
        raise ValueError('Repeated values are not allowed in regularization_dim')

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
