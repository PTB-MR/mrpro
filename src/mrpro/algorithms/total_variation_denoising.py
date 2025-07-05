"""Total Variation (TV) Denoising using PDHG."""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import torch

from mrpro.algorithms.optimizers.pdhg import pdhg
from mrpro.data.IData import IData
from mrpro.operators import LinearOperatorMatrix, ProximableFunctionalSeparableSum
from mrpro.operators.FiniteDifferenceOp import FiniteDifferenceOp
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared
from mrpro.operators.IdentityOp import IdentityOp
from mrpro.utils import unsqueeze_right


@overload
def total_variation_denoising(
    idata: IData,
    regularization_weights: Sequence[float] | Sequence[torch.Tensor],
    initial_image: torch.Tensor | None = None,
    max_iterations: int = 100,
    tolerance: float = 0,
) -> IData: ...


@overload
def total_variation_denoising(
    idata: torch.Tensor,
    regularization_weights: Sequence[float] | Sequence[torch.Tensor],
    initial_image: torch.Tensor | None = None,
    max_iterations: int = 100,
    tolerance: float = 0,
) -> torch.Tensor: ...


def total_variation_denoising(
    idata: IData | torch.Tensor,
    regularization_weights: float | Sequence[float] | Sequence[torch.Tensor],
    initial_image: torch.Tensor | None = None,
    max_iterations: int = 100,
    tolerance: float = 0,
) -> IData | torch.Tensor:
    r"""Apply total variation denoising.

    This algorithm solves the problem :math:`min_x \frac{1}{2}||x - y||_2^2 + \\sum_i l_i ||\nabla_i x||_1`
    by using the PDHG-algorithm. :math:`y` is the given noisy image, :math:`l_i` are the strengths of the regularization
    along the different dimensions and :math:`\nabla_i` is the finite difference operator applied to :math:`x` along
    different dimensions :math:`i`.

    Parameters
    ----------
    idata
        input image
    regularization_weights
        Strengths of the regularization (:math:`l_i`). Each entry is the regularization weight along a dimension of
        the reconstructed image starting at the back. E.g. (1,) will apply TV with l=1 along dimension (-1.).
        (3,0,2) will apply TV with l=2 along dimension (-1) and TV with l=3 along (-3). Single float will be applied
        along dimension -1.
    initial_image
        Initial image. If `None` then the target image :math:`y` will be used.
    max_iterations
        Maximum number of PDHG iterations.
    tolerance
            Tolerance of PDHG for relative change of the primal solution; if zero, `max_iterations` of PDHG are run.

    Returns
    -------
        the denoised image.
    """
    regularization_weights_ = torch.as_tensor(regularization_weights)
    img_tensor = idata if isinstance(idata, torch.Tensor) else idata.data

    l2_norm_squared = L2NormSquared(target=img_tensor)

    # TV regularization
    finite_difference_dim = [
        dim - len(regularization_weights_) for dim, weight in enumerate(regularization_weights_) if (weight != 0).any()
    ]
    nabla_operator = FiniteDifferenceOp(dim=finite_difference_dim, mode='forward')
    l1_norm = L1NormViewAsReal(weight=unsqueeze_right(regularization_weights_[finite_difference_dim], img_tensor.ndim))
    operator = LinearOperatorMatrix(((IdentityOp(),), (nabla_operator,)))

    initial_image = initial_image if initial_image is not None else img_tensor

    (img_tensor,) = pdhg(
        f=ProximableFunctionalSeparableSum(data_consistency, total_variation),
        g=None,
        operator=operator,
        initial_values=(initial_image,),
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return img_tensor if isinstance(idata, torch.Tensor) else IData(img_tensor, idata.header)
