"""Convolutional Dictionary based Image Denoising using FISTA."""

from __future__ import annotations

from typing import overload

import torch

from mrpro.algorithms.optimizers.cg import cg
from mrpro.algorithms.optimizers.pgd import pgd
from mrpro.data.IData import IData
from mrpro.operators.ConvSynthesisDictionaryOp import ConvSynthesisDictionaryOp
from mrpro.operators.FiniteDifferenceOp import FiniteDifferenceOp
from mrpro.operators.functionals import L1Norm, L2NormSquared
from mrpro.operators.IdentityOp import IdentityOp


@overload
def conv_synthesis_dictionary_denoising(
    idata: IData,
    kernel: torch.Tensor,
    low_pass_parameter: float,
    regularization_weight: float,
    initial_codes: torch.Tensor | None = None,
    max_iterations_low_pass_filtering: int = 16,
    max_iterations_pgd: int = 64,
    tolerance_low_pass_filtering: float = 1e-4,
) -> IData: ...


@overload
def conv_synthesis_dictionary_denoising(
    idata: torch.Tensor,
    kernel: torch.Tensor,
    low_pass_parameter: float,
    regularization_weight: float,
    initial_codes: torch.Tensor | None = None,
    max_iterations_low_pass_filtering: int = 16,
    max_iterations_pgd: int = 64,
    tolerance_low_pass_filtering: float = 1e-4,
) -> torch.Tensor: ...


def conv_synthesis_dictionary_denoising(
    idata: IData | torch.Tensor,
    kernel: torch.Tensor,
    low_pass_parameter: float,
    regularization_weight: float,
    initial_codes: torch.Tensor | None = None,
    max_iterations_low_pass_filtering: int = 16,
    max_iterations_pgd: int = 64,
    tolerance_low_pass_filtering: float = 1e-4,
) -> IData | torch.Tensor:
    r"""Apply image denoising using a pre-trained convolutional synthesis dictionary.

    This algorithm solves the problems
        :math:`x_{\mathrm{low}}:=\arg\min_x \frac{1}{2}||x - x||_2^2 + \frac{\beta}{2} ||\nabla x||_2^2`
        :math:`s^{\ast}:= \arg\min_s \frac{1}{2}||Ds - (y - x_{\mathrm{low}})||_2^2 + \lambda || s||_1`
        :math:`x^{\ast}:= Ds^\ast + x_{\mathrm{low}}`

    by using the FISTA-algorithm. :math:`y` is the given noisy image, :math:`\lambda` is the sparsity level
    :math:`\nabla` is the finite difference operator applied to :math:`x` along the last n dimensions that are
    defined by the number of dimensions that the convolutiona kernelis applied to.

    Parameters
    ----------
    idata
        noisy image
    kernel
        convolutional kernel for the synthesis operator
    low_pass_parameter
        regularization parameter of the
    regularization_weight
        Strength of the regularization, i.e. the sparsity of the coefficient maps.
    initial_codes
        initial estimate of the sparse coefficient maps; if `none`, it is initialized as zeros.
    max_iterations_low_pass_filtering
        maximum number of CG iterations for the low-pass filtering step.
    max_iterations_pgd
        maximum number of FISTA iterations for the sparse coding.
    tolerance_low_pass_filtering
        tolerance of CG for the relative change of the solution; if zero, `max_iterations` of CG are run.

    Returns
    -------
        the denoised image.
    """
    img_tensor = idata if isinstance(idata, torch.Tensor) else idata.data

    regularization_dimensions = tuple(-k for k in range(1, len(kernel.shape[1:]) + 1))[::-1]
    nabla_operator = FiniteDifferenceOp(dim=regularization_dimensions, mode='forward')

    (image_low_pass,) = cg(
        operator=IdentityOp() + low_pass_parameter * nabla_operator.gram,
        right_hand_side=img_tensor,
        initial_value=img_tensor,
        max_iterations=max_iterations_low_pass_filtering,
        tolerance=tolerance_low_pass_filtering,
    )

    conv_synthesis_operator = ConvSynthesisDictionaryOp(kernel=kernel, pad_mode='circular')
    l2_norm_squared = 0.5 * (L2NormSquared(target=img_tensor - image_low_pass) @ conv_synthesis_operator)

    l1_norm = regularization_weight * L1Norm()

    (initial_codes,) = (
        conv_synthesis_operator.H(torch.zeros_like(img_tensor)) if initial_codes is None else initial_codes
    )

    operator_norm = conv_synthesis_operator.operator_norm(
        torch.randn_like(initial_codes), dim=None, max_iterations=32
    ).item()
    stepsize = 0.97 / operator_norm**2
    (sparse_codes,) = pgd(
        f=l2_norm_squared,
        g=l1_norm,
        stepsize=stepsize,
        initial_value=initial_codes,
        max_iterations=max_iterations_pgd,
        convergent_iterates_variant=True,
    )

    img_tensor = conv_synthesis_operator(sparse_codes)[0] + image_low_pass

    return img_tensor if isinstance(idata, torch.Tensor) else IData(img_tensor, idata.header)
