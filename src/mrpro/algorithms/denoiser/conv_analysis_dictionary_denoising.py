"""Convolutional Analysis Dictionary based Image Denoising using PDHG."""

from __future__ import annotations

from typing import overload

import torch

from mrpro.algorithms.optimizers.pdhg import pdhg
from mrpro.data.IData import IData
from mrpro.operators import LinearOperatorMatrix, ProximableFunctionalSeparableSum
from mrpro.operators.ConvAnalysisDictionaryOp import ConvAnalysisDictionaryOp
from mrpro.operators.functionals import L1Norm, L2NormSquared
from mrpro.operators.IdentityOp import IdentityOp


@overload
def conv_analysis_dictionary_denoising(
    idata: IData,
    kernel: torch.Tensor,
    regularization_weight: float,
    initial_image: torch.Tensor | None = None,
    max_iterations_pdhg: int = 96,
    tolerance_pdhg: float = 1e-4,
) -> IData: ...


@overload
def conv_analysis_dictionary_denoising(
    idata: torch.Tensor,
    kernel: torch.Tensor,
    regularization_weight: float,
    initial_image: torch.Tensor | None = None,
    max_iterations_pdhg: int = 96,
    tolerance_pdhg: float = 1e-4,
) -> torch.Tensor: ...


def conv_analysis_dictionary_denoising(
    idata: IData | torch.Tensor,
    kernel: torch.Tensor,
    regularization_weight: float,
    initial_image: torch.Tensor | None = None,
    max_iterations_pdhg: int = 96,
    tolerance_pdhg: float = 1e-4,
) -> IData | torch.Tensor:
    r"""Apply image denoising using a given convolutional analysis dictionary.

    This algorithm solves the problem
        :math:`x^{\ast}:= \arg\min_x \frac{1}{2}||x - y||_2^2 + \lambda || H x||_1`,

    by using the primal dual hybrid gradient (PDHG) algorithm. Thereby, :math:`y` is the given noisy image,
    :math:`\lambda` is the sparsity level and :math:`H` a sparsifying convolutional analysis operator.

    Denoising is achieved by computing an image that is close to the noisy image but at the same time is sparse after
    the application of the convolutional filters. To solve the problem, the PDHG algorithm     is used.

    Parameters
    ----------
    idata
        noisy image
    kernel
        convolutional kernel for the convolutional analysis operator.
    regularization_weight
        strength of the regularization, i.e. the sparsity of the filtered images.
    initial_image
        initial estimate of the clean image; if `none`, it is initialized as the noisy image.
    max_iterations_pdhg
        maximum number of PDHG iterations to solve the problem.
    tolerance_pdhg
        tolerance of PDHG; if zero, `max_iterations` of PDHG are run.

    Returns
    -------
        the denoised image.
    """
    img_tensor = idata if isinstance(idata, torch.Tensor) else idata.data

    conv_analysis_operator = ConvAnalysisDictionaryOp(kernel=kernel, pad_mode='circular')
    l2_norm_squared = 0.5 * L2NormSquared(target=img_tensor)

    l1_norm = regularization_weight * L1Norm()
    operator = LinearOperatorMatrix(((IdentityOp(),), (conv_analysis_operator,)))

    initial_image = initial_image if initial_image is not None else img_tensor

    (img_tensor,) = pdhg(
        f=ProximableFunctionalSeparableSum(l2_norm_squared, l1_norm),
        g=None,
        operator=operator,
        initial_values=(initial_image,),
        max_iterations=max_iterations_pdhg,
        tolerance=tolerance_pdhg,
    )
    return img_tensor if isinstance(idata, torch.Tensor) else IData(img_tensor, idata.header)
