import torch
from torch.nn import Module, Parameter

from mrpro.data.KData import KData
from mrpro.operators.FourierOp import FourierOp
from mrpro.operators.IdentityOp import IdentityOp


class AnalyticCartesianDC(Module):
    r"""Analytic Cartesian data consistency.

    Solves the following problem:
    :math:`\min_x \|Ax - k\|_2^2 + \lambda \|x-p\|_2^2`
    where :math:`A` is the acquisition operator and :math:`k` is the data, :math:`\lambda` is the regularization parameter,
    and :math:`p` is the regularization image/prior analytically. :math:`A^H A` has to be diagonal. This is a special case
    for a Cartesian acquisition without coil sensitivity weighting. This can be used for either single-coil data or
    to apply data consistency to each coil image [NOSENSE]_

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
        super().__init__()
        self.regularization_weight = Parameter(torch.as_tensor(initial_regularization_weight))

    def forward(
        self,
        x: torch.Tensor,
        data: KData | torch.Tensor,
        fourier_op: FourierOp,
    ):
        if not isinstance(fourier_op, FourierOp) or fourier_op._nufft_dims or fourier_op._fast_fourier_op is None:
            raise ValueError('Only Cartesian acquisitions are supported')

        data_ = data.data if isinstance(data, KData) else data
        fft_op = fourier_op._fast_fourier_op
        sampling_op = fourier_op._cart_sampling_op if fourier_op._cart_sampling_op is not None else IdentityOp()
        (zero_filled,) = sampling_op.adjoint(data_)
        (k_pred,) = fft_op(x)
        (k,) = sampling_op.gram((zero_filled - k_pred) / (1 + self.regularization_weight))
        (delta,) = fft_op.H(k)
        return x + delta
