"""Total Variation (TV) Denoising using PDHG."""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import torch

from mr2.algorithms.optimizers.pdhg import pdhg
from mr2.data.IData import IData
from mr2.operators import LinearOperatorMatrix, ProximableFunctionalSeparableSum
from mr2.operators.FiniteDifferenceOp import FiniteDifferenceOp
from mr2.operators.functionals import L1NormViewAsReal, L2NormSquared
from mr2.operators.IdentityOp import IdentityOp
from mr2.utils import normalize_index, unsqueeze_right


@overload
def total_variation_denoising(
    idata: IData,
    regularization_dim: Sequence[int],
    regularization_weight: float | Sequence[float] | Sequence[torch.Tensor],
    initial_image: torch.Tensor | None = None,
    max_iterations: int = 100,
    tolerance: float = 0,
) -> IData: ...


@overload
def total_variation_denoising(
    idata: torch.Tensor,
    regularization_dim: Sequence[int],
    regularization_weight: float | Sequence[float] | Sequence[torch.Tensor],
    initial_image: torch.Tensor | None = None,
    max_iterations: int = 100,
    tolerance: float = 0,
) -> torch.Tensor: ...


def total_variation_denoising(
    idata: IData | torch.Tensor,
    regularization_dim: Sequence[int],
    regularization_weight: float | Sequence[float] | Sequence[torch.Tensor],
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
    regularization_dim
        Dimensions along which the total variation reguarization is applied (:math:`i`).
    regularization_weight
        Strengths of the regularization (:math:`l_i`). If a single values is given, it is applied to all dimensions.
        If a sequence is given, it must have the same length as `regularization_dim`.
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
    regularization_weight_ = torch.as_tensor(
        regularization_weight
        if isinstance(regularization_weight, Sequence)
        else [regularization_weight] * len(regularization_dim)
    )
    img_tensor = idata if isinstance(idata, torch.Tensor) else idata.data

    if len(regularization_dim) != len(regularization_weight_):
        raise ValueError('Regularization dimensions and weights must have the same length')
    regularization_dim = tuple(normalize_index(img_tensor.ndim, idx) for idx in regularization_dim)
    if len(regularization_dim) != len(set(regularization_dim)):
        raise ValueError('Repeated values are not allowed in regularization_dim')

    l2_norm_squared = L2NormSquared(target=img_tensor)

    # TV regularization
    nabla_operator = FiniteDifferenceOp(dim=regularization_dim, mode='forward')
    l1_norm = L1NormViewAsReal(weight=unsqueeze_right(regularization_weight_, img_tensor.ndim))
    operator = LinearOperatorMatrix(((IdentityOp(),), (nabla_operator,)))

    initial_image = initial_image if initial_image is not None else img_tensor

    (img_tensor,) = pdhg(
        f=ProximableFunctionalSeparableSum(l2_norm_squared, l1_norm),
        g=None,
        operator=operator,
        initial_values=(initial_image,),
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return img_tensor if isinstance(idata, torch.Tensor) else IData(img_tensor, idata.header)
