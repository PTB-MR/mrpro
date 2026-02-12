"""Total Variation (TV)-Regularized Reconstruction using PDHG."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from mr2.algorithms.optimizers.pdhg import pdhg
from mr2.algorithms.prewhiten_kspace import prewhiten_kspace
from mr2.algorithms.reconstruction.DirectReconstruction import DirectReconstruction
from mr2.data.CsmData import CsmData
from mr2.data.DcfData import DcfData
from mr2.data.IData import IData
from mr2.data.KData import KData
from mr2.data.KNoise import KNoise
from mr2.operators import LinearOperatorMatrix, ProximableFunctionalSeparableSum
from mr2.operators.DensityCompensationOp import DensityCompensationOp
from mr2.operators.FiniteDifferenceOp import FiniteDifferenceOp
from mr2.operators.functionals import L1NormViewAsReal, L2NormSquared
from mr2.operators.LinearOperator import LinearOperator
from mr2.operators.SensitivityOp import SensitivityOp
from mr2.utils import normalize_index, unsqueeze_right


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

    regularization_dim: Sequence[int]
    """Dimensions along which the total variation reguarization is applied :math:`i`."""

    regularization_weight: torch.Tensor
    """Strengths of the regularization along different dimensions :math:`l_i`."""

    def __init__(
        self,
        kdata: KData | None = None,
        fourier_op: LinearOperator | None = None,
        csm: Callable | CsmData | SensitivityOp | None = CsmData.from_idata_walsh,
        noise: KNoise | None = None,
        dcf: DcfData | DensityCompensationOp | None = None,
        *,
        max_iterations: int = 100,
        tolerance: float = 0,
        regularization_dim: Sequence[int],
        regularization_weight: float | Sequence[float] | Sequence[torch.Tensor],
    ) -> None:
        """Initialize TotalVariationRegularizedReconstruction.

        Parameters
        ----------
        kdata
            KData. If `kdata` is provided and `fourier_op` or `dcf` are `None`, then `fourier_op` and `dcf` are
            estimated based on `kdata`. Otherwise `fourier_op` and `dcf` are used as provided.
        fourier_op
            Instance of the `~mr2.operators.FourierOp` used for reconstruction. If `None`, set up based on `kdata`.
        csm
            Sensitivity maps for coil combination. If `None`, no coil combination is carried out, i.e. images for each
            coil are returned. If a `Callable` is provided, coil images are reconstructed using the adjoint of the
            `~mr2.operators.FourierOp` (including density compensation) and then sensitivity maps are calculated
            using the `Callable`. For this, `kdata` needs also to be provided.
            For examples have a look at the `mr2.data.CsmData` class e.g. `~mr2.data.CsmData.from_idata_walsh`
            or `~mr2.data.CsmData.from_idata_inati`.
        noise
            KNoise used for prewhitening. If `None`, no prewhitening is performed
        dcf
            K-space sampling density compensation. If `None`, set up based on `kdata`. The `dcf` is only used to
            calculate a starting estimate for PDHG.
        max_iterations
            Maximum number of PDHG iterations
        tolerance
            Tolerance of PDHG for relative change of the primal solution; if zero, `max_iterations` of PDHG are run.
        regularization_dim
            Dimensions along which the total variation reguarization is applied (:math:`i`).
        regularization_weight
            Strengths of the regularization (:math:`l_i`). If a single values is given, it is applied to all dimensions.
            If a sequence is given, it must have the same length as `regularization_dim`.

        Raises
        ------
        ValueError
            If the `kdata` and `fourier_op` are `None` or if `csm` is a `Callable` but `kdata` is `None`.
        ValueError
            If `regularization_dim` contains repeated values.
        ValueError
            If the length of `regularization_dim` and `regularization_weight` do not match
        """
        super().__init__(kdata, fourier_op, csm, noise, dcf)
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        if len(regularization_dim) != len(set(regularization_dim)):
            raise ValueError('Repeated values are not allowed in regularization_dim')
        self.regularization_dim = regularization_dim

        if isinstance(regularization_weight, float):
            regularization_weight = [regularization_weight] * len(regularization_dim)
        if len(regularization_dim) != len(regularization_weight):
            raise ValueError('Regularization dimensions and weights must have the same length')
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
        regularization_dim = tuple(normalize_index(kdata.ndim, idx) for idx in self.regularization_dim)
        if len(regularization_dim) != len(set(regularization_dim)):
            raise ValueError('Repeated values are not allowed in regularization_dim')

        if self.noise is not None:
            kdata = prewhiten_kspace(kdata, self.noise)

        acquisition_operator = self.fourier_op @ self.csm_op if self.csm_op is not None else self.fourier_op
        l2_norm_squared = L2NormSquared(target=kdata.data)

        # TV regularization
        nabla_operator = FiniteDifferenceOp(dim=regularization_dim, mode='forward')
        l1_norm = L1NormViewAsReal(weight=unsqueeze_right(self.regularization_weight, kdata.data.ndim))
        operator = LinearOperatorMatrix(((acquisition_operator,), (nabla_operator,)))

        initial_value = acquisition_operator.H(self.dcf_op(kdata.data)[0] if self.dcf_op is not None else kdata.data)[0]

        (img_tensor,) = pdhg(
            f=ProximableFunctionalSeparableSum(l2_norm_squared, l1_norm),
            g=None,
            operator=operator,
            initial_values=(initial_value,),
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )
        img = IData.from_tensor_and_kheader(img_tensor, kdata.header)
        return img
