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
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.utils import unsqueeze_right


class TotalVariationRegularizedReconstruction(DirectReconstruction):
    r"""TV-regularized reconstruction.

    This algorithm solves the problem :math:`min_x \frac{1}{2}||Ax - y||_2^2 + \sum_i l_i ||\nabla_i x||_1`
    by using the PDHG-algorithm. :math:`A` is the acquisition model (coil sensitivity maps, Fourier operator,
    k-space sampling), :math:`y` is the acquired k-space data, :math:`l_i` are the strengths of the regularization
    along the different dimensions and :math:`\nabla_i` is the finite difference operator applied to :math:`x` along
    different dimensions :math:`i`.
    """

    max_iterations: int
    """Maximum number of PDHG iterations."""

    tolerance: float
    """Tolerance of PDHG for relative change of the primal solution."""

    regularization_weights: torch.Tensor
    """Strengths of the regularization along different dimensions :math:`l_i`."""

    def __init__(
        self,
        kdata: KData | None = None,
        fourier_op: LinearOperator | None = None,
        csm: Callable | CsmData | None = CsmData.from_idata_walsh,
        noise: KNoise | None = None,
        dcf: DcfData | None = None,
        *,
        max_iterations: int = 100,
        tolerance: float = 0,
        regularization_weights: float | Sequence[float] | Sequence[torch.Tensor],
    ) -> None:
        """Initialize TotalVariationRegularizedReconstruction.

        Parameters
        ----------
        kdata
            KData. If `kdata` is provided and `fourier_op` or `dcf` are `None`, then `fourier_op` and `dcf` are
            estimated based on `kdata`. Otherwise `fourier_op` and `dcf` are used as provided.
        fourier_op
            Instance of the `~mrpro.operators.FourierOp` used for reconstruction. If `None`, set up based on `kdata`.
        csm
            Sensitivity maps for coil combination. If `None`, no coil combination is carried out, i.e. images for each
            coil are returned. If a `Callable` is provided, coil images are reconstructed using the adjoint of the
            `~mrpro.operators.FourierOp` (including density compensation) and then sensitivity maps are calculated
            using the `Callable`. For this, `kdata` needs also to be provided.
            For examples have a look at the `mrpro.data.CsmData` class e.g. `~mrpro.data.CsmData.from_idata_walsh`
            or `~mrpro.data.CsmData.from_idata_inati`.
        noise
            KNoise used for prewhitening. If `None`, no prewhitening is performed
        dcf
            K-space sampling density compensation. If `None`, set up based on `kdata`. The `dcf` is only used to
            calculate a starting estimate for PDHG.
        max_iterations
            Maximum number of PDHG iterations
        tolerance
            Tolerance of PDHG for relative change of the primal solution; if zero, `max_iterations` of PDHG are run.
        regularization_weights
            Strengths of the regularization (:math:`l_i`). Each entry is the regularization weight along a dimension of
            the reconstructed image starting at the back. E.g. (1,) will apply TV with l=1 along dimension (-1,).
            (3,0,2) will apply TV with l=2 along dimension (-1) and TV with l=3 along (-3). Single float will be applied
            along dimension -1.

        Raises
        ------
        ValueError
            If the `kdata` and `fourier_op` are `None` or if `csm` is a `Callable` but `kdata` is `None`.
        """
        super().__init__(kdata, fourier_op, csm, noise, dcf)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization_weights = torch.as_tensor(regularization_weights)

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

        acquisition_operator = self.fourier_op @ self.csm.as_operator() if self.csm is not None else self.fourier_op
        data_consistency = 0.5 * L2NormSquared(target=kdata.data)

        # TV regularization
        finite_difference_dim = [
            dim - len(self.regularization_weights)
            for dim, weight in enumerate(self.regularization_weights)
            if weight != 0
        ]
        nabla_operator = FiniteDifferenceOp(dim=finite_difference_dim, mode='forward')
        total_variation = L1NormViewAsReal(
            weight=unsqueeze_right(self.regularization_weights[finite_difference_dim], kdata.data.ndim)
        )
        operator = LinearOperatorMatrix(((acquisition_operator,), (nabla_operator,)))

        initial_value = acquisition_operator.H(
            self.dcf.as_operator()(kdata.data)[0] if self.dcf is not None else kdata.data
        )[0]

        (img_tensor,) = pdhg(
            f=ProximableFunctionalSeparableSum(data_consistency, total_variation),
            g=None,
            operator=operator,
            initial_values=(initial_value,),
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )
        img = IData.from_tensor_and_kheader(img_tensor, kdata.header)
        return img
