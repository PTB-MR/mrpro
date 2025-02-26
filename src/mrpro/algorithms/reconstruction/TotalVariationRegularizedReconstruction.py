"""Total Variation (TV)-Regularized Reconstruction using PDHG."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from mrpro.algorithms.optimizers.pdhg import pdhg
from mrpro.algorithms.prewhiten_kspace import prewhiten_kspace
from mrpro.algorithms.reconstruction.DirectReconstruction import DirectReconstruction
from mrpro.data.CsmData import CsmData
from mrpro.data.DcfData import DcfData
from mrpro.data.IData import IData
from mrpro.data.KData import KData
from mrpro.data.KNoise import KNoise
from mrpro.operators import LinearOperatorMatrix, ProximableFunctionalSeparableSum
from mrpro.operators.FiniteDifferenceOp import FiniteDifferenceOp
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared, ZeroFunctional
from mrpro.operators.LinearOperator import LinearOperator


class TotalVariationRegularizedReconstruction(DirectReconstruction):
    r"""TV-regularized reconstruction.

    This algorithm solves the problem :math:`min_x \frac{1}{2}||(Ax - y)||_2^2 + ||L\nabla x||_1`
    by using the PDHG-algorithm. :math:`A` is the acquisition model (coil sensitivity maps, Fourier operator,
    k-space sampling), :math:`y` is the acquired k-space data, :math:`L` is the strength of the regularization and
    :math:`\nabla` is the finite difference operator applied to :math:`x`.
    """

    n_iterations: int
    """Number of PDHG iterations."""

    regularization_weight: torch.Tensor
    """Strength of the regularization :math:`L`."""

    def __init__(
        self,
        kdata: KData | None = None,
        fourier_op: LinearOperator | None = None,
        csm: Callable | CsmData | None = CsmData.from_idata_walsh,
        noise: KNoise | None = None,
        dcf: DcfData | None = None,
        *,
        n_iterations: int = 20,
        regularization_weight: Sequence[float] | Sequence[torch.Tensor],
    ) -> None:
        """Initialize TotalVariationRegularizedReconstruction.

        Parameters
        ----------
        kdata
            KData. If kdata is provided and fourier_op or dcf are None, then fourier_op and dcf are estimated based on
            kdata. Otherwise fourier_op and dcf are used as provided.
        fourier_op
            Instance of the FourierOperator used for reconstruction. If None, set up based on kdata.
        csm
            Sensitivity maps for coil combination. If None, no coil combination is carried out, i.e. images for each
            coil are returned. If a callable is provided, coil images are reconstructed using the adjoint of the
            FourierOperator (including density compensation) and then sensitivity maps are calculated using the
            callable. For this, kdata needs also to be provided. For examples have a look at the CsmData class
            e.g. from_idata_walsh or from_idata_inati.
        noise
            KNoise used for prewhitening. If None, no prewhitening is performed
        dcf
            K-space sampling density compensation. If None, set up based on kdata. The dcf is only used to calculate a
            starting estimate for PDHG.
        n_iterations
            Number of PDHG iterations
        regularization_weight
            Strength of the regularization (:math:`L`). Each entry is the regularization weight along a dimension of
            the reconstructed image starting at the back. E.g. (1,) will apply TV with L=1 along dimension (-1,).
            (3,0,2) will apply TV with L=2 along dimension (-1) and TV with L=3 along (-3).

        Raises
        ------
        ValueError
            If the kdata and fourier_op are None or if csm is a Callable but kdata is None.
        """
        super().__init__(kdata, fourier_op, csm, noise, dcf)
        self.n_iterations = n_iterations
        self.regularization_weight = torch.as_tensor(regularization_weight)

    def forward(self, kdata: KData) -> IData:
        """Apply the reconstruction.

        Parameters
        ----------
        kdata
            k-space data to reconstruct.

        Returns
        -------
            the reconstruced image.
        """
        if self.noise is not None:
            kdata = prewhiten_kspace(kdata, self.noise)

        # Create the acquisition model A = F S if the CSM S is defined otherwise A = F with the Fourier operator F
        acquisition_operator = self.fourier_op @ self.csm.as_operator() if self.csm is not None else self.fourier_op

        # L2-norm for the data consistency term
        l2 = 0.5 * L2NormSquared(target=kdata.data, divide_by_n=False)

        # Finite difference operator and corresponding L1-norm
        nabla_operator = [
            (FiniteDifferenceOp(dim=(dim - len(self.regularization_weight),), mode='forward'),)
            for dim, weight in enumerate(self.regularization_weight)
            if weight != 0
        ]
        l1 = [weight * L1NormViewAsReal(divide_by_n=False) for weight in self.regularization_weight if weight != 0]

        f = ProximableFunctionalSeparableSum(l2, *l1)
        g = ZeroFunctional()
        operator = LinearOperatorMatrix(((acquisition_operator,), *nabla_operator))

        # Initial value
        initial_value = acquisition_operator.H(
            self.dcf.as_operator()(kdata.data)[0] if self.dcf is not None else kdata.data
        )[0]

        (img_tensor,) = pdhg(
            f=f, g=g, operator=operator, initial_values=(initial_value,), max_iterations=self.n_iterations
        )
        img = IData.from_tensor_and_kheader(img_tensor, kdata.header)
        return img
