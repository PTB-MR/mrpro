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
    r"""Plug-and-Play (PnP) priors reconstruction.

    This algorithm solves the problem :math:`min_x \frac{1}{2}||Ax - y||_2^2 + \beta s(x)` using the plug-and-play
    ADMM scheme of [Venkatakrishnan2013]_. :math:`A` is the acquisition model (coil sensitivity maps, Fourier operator,
    k-space sampling), :math:`y` is the acquired k-space data and :math:`s` is an (implicit) regularizer. Using the
    splitting :math:`x = v`, each ADMM iteration consists of a data-consistency step

    :math:`x_{k+1} = argmin_x \frac{1}{2}||Ax - y||_2^2 + \frac{\lambda}{2}||x - (v_k - u_k)||_2^2`,

    which is solved with CG, a denoising step :math:`v_{k+1} = H(x_{k+1} + u_k)` in which the proximal operator of
    :math:`\beta s` is replaced by an arbitrary denoiser :math:`H`, and the update of the scaled dual variable
    :math:`u_{k+1} = u_k + x_{k+1} - v_{k+1}`. :math:`\lambda` is the ADMM penalty parameter.

    References
    ----------
    .. [Venkatakrishnan2013] Venkatakrishnan, S. V., Bouman, C. A., & Wohlberg, B. (2013). Plug-and-Play priors for
       model based reconstruction. IEEE Global Conference on Signal and Information Processing, 945-948.
       https://doi.org/10.1109/GlobalSIP.2013.6737048
    """

    denoiser: Callable[[torch.Tensor], torch.Tensor]
    """Denoiser :math:`H` applied to the image tensor, returning a tensor of the same shape."""

    admm_regularization_strength: float
    r"""ADMM penalty parameter :math:`\lambda`."""

    max_iterations: int
    """Maximum number of ADMM iterations."""

    max_iterations_cg: int
    """Maximum number of iterations of internal CG."""

    tolerance: float
    """Tolerance for the relative change of the image between two ADMM iterations."""

    tolerance_cg: float
    """Tolerance for the residual norm of the internal CG."""

    def __init__(
        self,
        kdata: KData | None = None,
        fourier_op: LinearOperator | None = None,
        csm: Callable | CsmData | SensitivityOp | None = CsmData.from_idata_walsh,
        noise: KNoise | None = None,
        dcf: DcfData | DensityCompensationOp | None = None,
        *,
        denoiser: Callable[[torch.Tensor], torch.Tensor],
        admm_regularization_strength: float,
        max_iterations: int = 100,
        max_iterations_cg: int = 100,
        tolerance: float = 0,
        tolerance_cg: float = 1e-6,
    ) -> None:
        r"""Initialize PlugAndPlayPriorsReconstruction.

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
            calculate a starting estimate.
        denoiser
            Denoiser :math:`H` applied to the image tensor, returning a tensor of the same shape.
        admm_regularization_strength
            ADMM penalty parameter :math:`\lambda`.
        max_iterations
            Maximum number of ADMM iterations.
        max_iterations_cg
            Maximum number of iterations of internal CG.
        tolerance
            Tolerance for the relative change of the image between two ADMM iterations; if zero, `max_iterations`
            iterations are run.
        tolerance_cg
            Tolerance for the residual norm of the internal CG.

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

        acquisition_operator = self.fourier_op @ self.csm_op if self.csm_op is not None else self.fourier_op
        (right_hand_side,) = acquisition_operator.H(kdata.data)
        operator = acquisition_operator.gram + self.admm_regularization_strength * IdentityOp()

        image = acquisition_operator.H(self.dcf_op(kdata.data)[0] if self.dcf_op is not None else kdata.data)[0]

        denoised_image = image
        dual_variable = torch.zeros_like(image)

        for _ in range(self.max_iterations):
            image_old = image
            (image,) = cg(
                operator=operator,
                right_hand_side=right_hand_side + self.admm_regularization_strength * (denoised_image - dual_variable),
                initial_value=image_old,
                max_iterations=self.max_iterations_cg,
                tolerance=self.tolerance_cg,
            )
            denoised_image = self.denoiser(image + dual_variable)
            dual_variable = dual_variable + image - denoised_image

            if self.tolerance != 0:
                relative_change = torch.linalg.vector_norm(image - image_old) / torch.linalg.vector_norm(image_old)
                if relative_change < self.tolerance:
                    break

        return IData.from_tensor_and_kheader(image, kdata.header)
