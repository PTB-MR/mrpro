"""Analytic Cartesian data consistency."""

from typing import overload

import torch
from torch.nn import Module, Parameter

from mr2.data.KData import KData
from mr2.operators.FourierOp import FourierOp
from mr2.operators.IdentityOp import IdentityOp


class AnalyticCartesianDC(Module):
    r"""Analytic Cartesian data consistency.

    Solves the following problem:
    :math:`\min_x \|Ax - k\|_2^2 + \lambda \|x-p\|_2^2`
    where :math:`A` is the acquisition operator and :math:`k` is the data, :math:`\lambda` is the regularization
    parameter and :math:`p` is the regularization image/prior analytically. :math:`A^H A` has to be diagonal. This is a
    special case for a Cartesian acquisition without coil sensitivity weighting. This can be used for either single-coil
    data or to apply data consistency to each coil image [NOSENSE]_.

    References
    ----------
    .. [NOSENSE] Zimmermann, FF, and Kofler, Andreas. "NoSENSE: Learned unrolled cardiac MRI reconstruction without
        explicit sensitivity maps." STACOM@MICCAI 2023. https://arxiv.org/abs/2309.15608

    Parameters
    ----------
    initial_regularization_weight
        Initial regularization weight. The regularization weight is a trainable parameter.


    """

    def __init__(self, initial_regularization_weight: torch.Tensor | float):
        """Initialize the data consistency.

        Parameters
        ----------
        initial_regularization_weight
            Initial regularization weight. The regularization weight is a trainable parameter.
            Must be a positive scalar.
        """
        super().__init__()
        weight = torch.as_tensor(initial_regularization_weight)
        if weight.ndim != 0:
            raise ValueError('Regularization weight must be a scalar')
        if weight.item() <= 0:
            raise ValueError('Regularization weight must be positive')
        self.log_weight = Parameter(weight.log())

    @overload
    def __call__(self, image: torch.Tensor, data: KData, fourier_op: FourierOp | None = None) -> torch.Tensor: ...

    @overload
    def __call__(self, image: torch.Tensor, data: torch.Tensor, fourier_op: FourierOp) -> torch.Tensor: ...

    def __call__(
        self, image: torch.Tensor, data: KData | torch.Tensor, fourier_op: FourierOp | None = None
    ) -> torch.Tensor:
        """Apply the data consistency.

        Parameters
        ----------
        image
            Current image estimate, i.e. the regularized image.
        data
            k-space data.
        fourier_op
            Fourier operator matching the k-space data. If None and data is provided as a `~mr2.data.KData` object,
            the Fourier operator is automatically created from the data.

        Returns
        -------
            Updated image estimate.
        """
        return super().__call__(image, data, fourier_op)

    def forward(
        self, image: torch.Tensor, data: KData | torch.Tensor, fourier_op: FourierOp | None = None
    ) -> torch.Tensor:
        """Apply the data consistency."""
        if fourier_op is None:
            if isinstance(data, KData):
                fourier_op = FourierOp.from_kdata(data)
            else:
                raise ValueError('Either a KData or a FourierOp is required')

        if not isinstance(fourier_op, FourierOp) or fourier_op._nufft_dims or fourier_op._fast_fourier_op is None:
            raise ValueError('Only Cartesian acquisitions are supported')

        data_ = data.data if isinstance(data, KData) else data
        fft_op = fourier_op._fast_fourier_op
        sampling_op = fourier_op._cart_sampling_op if fourier_op._cart_sampling_op is not None else IdentityOp()
        (zero_filled,) = sampling_op.adjoint(data_)
        (k_pred,) = fft_op(image)
        regularization_weight = self.log_weight.exp()
        (k,) = sampling_op.gram((zero_filled - k_pred) / (1 + regularization_weight))
        (delta,) = fft_op.H(k)
        return image + delta
