"""Patch-based Dictionary Denoising using Proximal Gradient Descent (PGD)."""

# %%
from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import torch

from mrpro.algorithms.optimizers import pgd
from mrpro.data.IData import IData
from mrpro.operators import LinearOperator, PatchOp
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared


@overload
def patch_based_denoising(
    idata: IData,
    dictionary_op: LinearOperator,
    patch_dim: Sequence[int],
    patch_size: Sequence[int],
    stride: Sequence[int] | None = None,
    regularization_weight: float = 1.0,
    max_iterations: int = 100,
    backtrack_factor: float = 1.0,
    convergent_iterates_variant: bool = False,
) -> IData: ...


@overload
def patch_based_denoising(
    idata: torch.Tensor,
    dictionary_op: LinearOperator,
    patch_dim: Sequence[int],
    patch_size: Sequence[int],
    stride: Sequence[int] | None = None,
    regularization_weight: float = 1.0,
    max_iterations: int = 100,
    backtrack_factor: float = 1.0,
    convergent_iterates_variant: bool = False,
) -> torch.Tensor: ...


def patch_based_denoising(
    idata: IData | torch.Tensor,
    dictionary_op: LinearOperator,
    patch_dim: Sequence[int],
    patch_size: Sequence[int],
    stride: Sequence[int] | None = None,
    regularization_weight: float = 1.0,
    max_iterations: int = 100,
    backtrack_factor: float = 1.0,
    convergent_iterates_variant: bool = False,
) -> IData | torch.Tensor:
    r"""Apply patch-based dictionary denoising.

    The noisy image :math:`x_\mathrm{noisy}` is split into (possibly overlapping) patches, defined by
    the patch size and strides. Each patch is sparsely coded with respect to a dictionary :math:`\Psi`:

    .. math::

        \mu_j &:= \mathrm{Mean}(R_j x_\mathrm{noisy}) \\
        z_j &:= R_j x_\mathrm{noisy} - \mu_j \\
        \gamma_j^* &:= \arg\min_{\gamma_j} \frac{1}{2} \| \Psi \gamma_j - z_j \|_2^2 + \lambda \| \gamma_j \|_1 \\
        x_\mathrm{Rec} &:= \frac{1}{d} \sum_j R_j^T \left( \Psi \gamma_j^* + \mu_j \right)

    where :math:`R_j` extracts the :math:`j`-th patch from the image and the adjoint :math:`R_j^T` puts it back in its
    position in the image. :math:`\mu_j` is the mean of the :math:`j`-th patch and
    :math:`z_j` is the corresponding patch
    with mean zero. :math:`\lambda > 0` is the regularization weight for the sparse coefficients.
    :math:`d` counts for the overlap of the patches.
    The minimization problem for sparse coefficients :math:`\gamma_j^*` is solved
    with the Proximal Gradient Descent (PGD) algorithm.

    Parameters
    ----------
    idata
        input image
    dictionary_op
        Linear Operator representing the dictionary (:math:`\Psi`), mapping the sparse coefficients
        :math:`\gamma_j` to the patches.
    patch_dim
        Dimension(s) to extract patches from.
    patch_size
        Size of patches.
    stride
        Stride for extracting the patches.
    regularization_weight
        Strength of the regularization (:math:`\lambda`).
    max_iterations
        Maximum number of PGD iterations.
    backtrack_factor
        Factor for backtracking line search. If 1.0, no backtracking is performed.
    convergent_iterates_variant
        If `True`, use the convergent iterates variant of PGD.

    Returns
    -------
        the denoised image.
    """
    img_tensor = idata if isinstance(idata, torch.Tensor) else idata.data
    patch_op = PatchOp(
        patch_size=patch_size,
        stride=stride,
        dim=patch_dim,
    )

    # extract patches: (n_patches, ..., patch_size)
    (patches,) = patch_op(img_tensor)

    # remove the mean of each patch
    mean_dims = tuple(d - img_tensor.ndim if d >= 0 else d for d in patch_dim)
    patches_average = patches.mean(dim=mean_dims, keepdim=True)
    patches_zero_mean = patches - patches_average

    # compute functionals f and g for pgd
    l2 = 0.5 * L2NormSquared(target=patches_zero_mean, divide_by_n=False)
    l1 = L1NormViewAsReal(divide_by_n=False)
    f = l2 @ dictionary_op
    g = regularization_weight * l1

    # initial coefficients: Psi^H (patches_zero_mean)
    (initial_coefficients,) = dictionary_op.adjoint(patches_zero_mean)

    # compute stepsize for pgd from the operator norm of the dictionary
    op_norm = dictionary_op.operator_norm(initial_value=torch.randn_like(initial_coefficients), dim=None)
    stepsize = 1.0 / op_norm.item() ** 2

    # solve the problem for sparse coefficients
    (opt_sparse_coefficients,) = pgd(
        f=f,
        g=g,
        initial_value=initial_coefficients,
        stepsize=stepsize,
        max_iterations=max_iterations,
        backtrack_factor=backtrack_factor,
        convergent_iterates_variant=convergent_iterates_variant,
    )

    # compute the denoised patches, add the means back and put patches back into image
    (opt_patches,) = dictionary_op(opt_sparse_coefficients)
    (opt_img_tensor,) = patch_op.adjoint(opt_patches + patches_average)

    # average overlapping patches
    (ones_patches,) = patch_op(torch.ones_like(img_tensor))
    (overlap_map,) = patch_op.adjoint(ones_patches)
    opt_img_tensor = opt_img_tensor / overlap_map

    return opt_img_tensor if isinstance(idata, torch.Tensor) else IData(opt_img_tensor, idata.header)


# %%
