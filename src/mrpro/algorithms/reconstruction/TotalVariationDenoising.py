"""Total Variation (TV) Denoising using PDHG."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from mrpro.algorithms.optimizers.pdhg import pdhg
from mrpro.data.IData import IData
from mrpro.operators import LinearOperatorMatrix, ProximableFunctionalSeparableSum
from mrpro.operators.FiniteDifferenceOp import FiniteDifferenceOp
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared, ZeroFunctional
from mrpro.operators.IdentityOp import IdentityOp


class TotalVariationDenoising(torch.nn.Module):
    r"""TV denoising.

    This algorithm solves the problem :math:`min_x \frac{1}{2}||(x - y)||_2^2 + \sum_i l_i ||\nabla_i x||_1`
    by using the PDHG-algorithm. :math:`y` is the given noisy image, :math:`l_i` are the strengths of the regularization
    along the different dimensions and :math:`\nabla_i` is the finite difference operator applied to :math:`x` along
    different dimensions :math:`i`.
    """

    regularization_weight: torch.Tensor
    """Strength of the regularization :math:`L`."""

    initial_image: torch.Tensor | None
    """Initial image for PDHG-algorithm"""

    max_iterations: int
    """Maximum number of PDHG iterations."""

    def __init__(
        self,
        regularization_weights: Sequence[float] | Sequence[torch.Tensor],
        initial_image: torch.Tensor | None = None,
        max_iterations: int = 32,
    ) -> None:
        """Initialize TotalVariationDenoising.

        Parameters
        ----------
        regularization_weights
            Strengths of the regularization (:math:`l_i`). Each entry is the regularization weight along a dimension of
            the reconstructed image starting at the back. E.g. (1,) will apply TV with l=1 along dimension (-1,).
            (3,0,2) will apply TV with l=2 along dimension (-1) and TV with l=3 along (-3).
        initial_image
            Initial image. If None then the target image :math:`y` will be used.
        max_iterations
            Maximum number of PDHG iterations

        """
        super().__init__()
        self.max_iterations = max_iterations
        self.regularization_weights = torch.as_tensor(regularization_weights)
        self.initial_image = initial_image

    # Required for type hinting
    def __call__(self, idata: IData) -> IData:
        """Apply the reconstruction."""
        return super().__call__(idata)

    def forward(self, idata: IData) -> IData:
        """Apply the denoising.

        Parameters
        ----------
        idata
            input image

        Returns
        -------
            the denoised image.
        """
        # L2-norm for the data consistency term
        l2 = 0.5 * L2NormSquared(target=idata.data, divide_by_n=False)

        # Finite difference operator and corresponding L1-norm
        nabla_operator = [
            (FiniteDifferenceOp(dim=(dim - len(self.regularization_weights),), mode='forward'),)
            for dim, weight in enumerate(self.regularization_weights)
            if weight != 0
        ]
        l1 = [weight * L1NormViewAsReal(divide_by_n=False) for weight in self.regularization_weights if weight != 0]

        f = ProximableFunctionalSeparableSum(l2, *l1)
        g = ZeroFunctional()
        operator = LinearOperatorMatrix(((IdentityOp(),), *nabla_operator))

        initial_image = self.initial_image if self.initial_image is not None else idata.data

        (img_tensor,) = pdhg(
            f=f, g=g, operator=operator, initial_values=(initial_image,), max_iterations=self.max_iterations
        )
        return IData(img_tensor, idata.header)
