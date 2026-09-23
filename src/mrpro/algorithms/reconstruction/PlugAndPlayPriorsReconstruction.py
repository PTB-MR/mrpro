"""Plug and Play (PnP) Priors Reconstruction."""

from __future__ import annotations

from collections.abc import Callable

import torch

from mrpro.algorithms.optimizers import cg
from mrpro.algorithms.prewhiten_kspace import prewhiten_kspace
from mrpro.algorithms.reconstruction.DirectReconstruction import DirectReconstruction
from mrpro.data.CsmData import CsmData
from mrpro.data.DcfData import DcfData
from mrpro.data.IData import IData
from mrpro.data.KData import KData
from mrpro.data.KNoise import KNoise
from mrpro.operators.DensityCompensationOp import DensityCompensationOp
from mrpro.operators.IdentityOp import IdentityOp
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.SensitivityOp import SensitivityOp


class PlugAndPlayPriorsReconstruction(DirectReconstruction):
    r"""Plug-and-Play Priors reconstruction.

    This algorithm ...
    """

    denoiser: Callable
    """Denoiser function."""

    admm_regularization_strength: torch.Tensor
    """Strengths of the ADMM regularization."""

    max_iterations: int
    """Maximum number of iterations."""

    max_iterations_cg: int
    """Maximum number of iterations of internal CG."""

    tolerance: float
    """Tolerance for the convergence check."""

    tolerance_cg: float
    """Tolerance for the convergence check of the internal CG."""

    def __init__(
        self,
        kdata: KData | None = None,
        fourier_op: LinearOperator | None = None,
        csm: Callable | CsmData | SensitivityOp | None = CsmData.from_idata_walsh,
        noise: KNoise | None = None,
        dcf: DcfData | DensityCompensationOp | None = None,
        *,
        denoiser: Callable[[torch.Tensor], torch.Tensor],
        admm_regularization_strength: torch.Tensor,
        max_iterations: int = 100,
        max_iterations_cg: int = 100,
        tolerance: float = 0,
        tolerance_cg: float = 1e-6,
    ) -> None:
        """Initialize PlugAndPlayReconstruction.

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
        denoiser
            Denoiser function.
        admm_regularization_strength
            Strengths of the ADMM regularization.
        max_iterations
            Maximum number of PDHG iterations
        max_iterations_cg
            Maximum number of iterations of internal CG.
        tolerance
            Tolerance of PDHG for relative change of the primal solution; if zero, `max_iterations` of PDHG are run.
        tolerance_cg
            Tolerance for the convergence check of the internal CG.

        Raises
        ------
        ValueError
            If the `kdata` and `fourier_op` are `None` or if `csm` is a `Callable` but `kdata` is `None`.
        """
        super().__init__(kdata, fourier_op, csm, noise, dcf)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.tolerance_cg = tolerance_cg
        self.max_iterations_cg = max_iterations_cg
        self.denoiser = denoiser
        self.admm_regularization_strength = admm_regularization_strength

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

        acquisition_op = self.fourier_op
        if self.csm_op is not None:
            acquisition_op = acquisition_op @ self.csm_op

        (adjoint_op,) = acquisition_op.H(kdata.data)

        initial_image = acquisition_op.H(self.dcf_op(kdata.data)[0] if self.dcf_op is not None else kdata.data)[0]

        dual_variable = torch.zeros_like(initial_image)
        image_hat = torch.zeros_like(initial_image)

        for _iter in range(self.max_iterations):
            (image_hat,) = cg(
                operator=acquisition_op.H @ acquisition_op + self.admm_regularization_strength * IdentityOp(),
                right_hand_side=adjoint_op + self.admm_regularization_strength * (image_hat - dual_variable),
                initial_value=initial_image,
                max_iterations=self.max_iterations_cg,
            )

            denoise_image_hat = self.denoiser(image_hat + dual_variable)
            dual_variable = dual_variable + image_hat - denoise_image_hat

        return IData.from_tensor_and_kheader(denoise_image_hat, kdata.header)
