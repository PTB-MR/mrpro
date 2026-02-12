"""Regularized Iterative SENSE Reconstruction."""

from __future__ import annotations

from collections.abc import Callable

import torch

from mr2.algorithms.optimizers.cg import cg
from mr2.algorithms.prewhiten_kspace import prewhiten_kspace
from mr2.algorithms.reconstruction.DirectReconstruction import DirectReconstruction
from mr2.data.CsmData import CsmData
from mr2.data.DcfData import DcfData
from mr2.data.IData import IData
from mr2.data.KData import KData
from mr2.data.KNoise import KNoise
from mr2.operators.DensityCompensationOp import DensityCompensationOp
from mr2.operators.IdentityOp import IdentityOp
from mr2.operators.LinearOperator import LinearOperator
from mr2.operators.SensitivityOp import SensitivityOp
from mr2.utils import unsqueeze_right


class RegularizedIterativeSENSEReconstruction(DirectReconstruction):
    r"""Regularized iterative SENSE reconstruction.

    This algorithm solves the problem :math:`\min_{x} ||Ax - y||_2^2 + \lambda||Bx - x_{reg}||_2^2`
    by using a conjugate gradient algorithm to solve
    :math:`H x = b` with :math:`H = A^H A + \lambda B^H B` and :math:`b = A^H y + \lambda B^H x_{reg}` where :math:`A`
    is the acquisition model (coil sensitivity maps, Fourier operator, k-space sampling), :math:`y` is the acquired
    k-space data, :math:`\lambda` is the strength of the regularization, and :math:`x_{reg}` is the regularization image
    (i.e. a prior). :math:`B` is a linear operator applied to :math:`x`.
    """

    n_iterations: int
    """Number of CG iterations."""

    regularization_data: torch.Tensor
    r"""Regularization data (i.e. prior) :math:`x_{\mathrm{reg}}`."""

    regularization_weight: torch.Tensor
    r"""Strength of the regularization :math:`\lambda`."""

    regularization_op: LinearOperator
    """Linear operator :math:`B` applied to the current estimate in the regularization term."""

    def __init__(
        self,
        kdata: KData | None = None,
        fourier_op: LinearOperator | None = None,
        csm: Callable | CsmData | SensitivityOp | None = CsmData.from_idata_walsh,
        noise: KNoise | None = None,
        dcf: DcfData | DensityCompensationOp | None = None,
        *,
        n_iterations: int = 5,
        regularization_data: float | torch.Tensor = 0.0,
        regularization_weight: float | torch.Tensor,
        regularization_op: LinearOperator | None = None,
    ) -> None:
        r"""Initialize RegularizedIterativeSENSEReconstruction.

        For a unregularized version of the iterative SENSE algorithm the regularization_weight can be set to ``0``
        or `~mr2.algorithms.reconstruction.IterativeSENSEReconstruction` algorithm can be used.

        Parameters
        ----------
        kdata
            If `kdata` is provided and `fourier_op` or `dcf` are `None`, then `fourier_op` and `dcf` are estimated
            based on `kdata`. Otherwise `fourier_op` and `dcf` are used as provided.
        fourier_op
            Instance of the `~mr2.operators.FourierOp` used for reconstruction.
            If `None`, set up based on `kdata`.
        csm
            Sensitivity maps for coil combination. If `None`, no coil combination is carried out, i.e. images for each
            coil are returned. If a `Callable` is provided, coil images are reconstructed using the adjoint of the
            `~mr2.operators.FourierOp` (including density compensation) and then sensitivity maps are calculated
            using the `Callable`. For this, `kdata` needs also to be provided.
            For examples have a look at the `mr2.data.CsmData` class e.g. `~mr2.data.CsmData.from_idata_walsh`
            or `~mr2.data.CsmData.from_idata_inati`.
        noise
            Noise used for prewhitening. If `None`, no prewhitening is performed
        dcf
            K-space sampling density compensation. If `None`, set up based on `kdata`.
            Used to obtain a the starting point for the CG algorithm as the scaled density compensated direct
            reconstruction [FESSLER2010]_.
        n_iterations
            Number of CG iterations
        regularization_data
            Regularization data, e.g. a reference image (:math:`x_0`).
        regularization_weight
            Strength of the regularization (:math:`\lambda`).
        regularization_op
            Linear operator :math:`B` applied to the current estimate in the regularization term. If None, nothing is
            applied to the current estimate.

        References
        ----------
        .. [FESSLER2010] Fessler, J.A., Noll, D.C: Iterative Reconstruction Methods for Non-Cartesian MRI.
           https://ece-classes.usc.edu/ee591/library/Fessler-Iterative%20Reconstruction.pdf

        Raises
        ------
        `ValueError`
            If the `kdata` and `fourier_op` are `None` or if `csm` is a `Callable` but `kdata` is None.
        """
        super().__init__(kdata, fourier_op, csm, noise, dcf)
        self.n_iterations = n_iterations
        self.regularization_data = torch.as_tensor(regularization_data)
        self.regularization_weight = torch.as_tensor(regularization_weight)
        self.regularization_op = regularization_op if regularization_op is not None else IdentityOp()

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

        acquisition_model = self.fourier_op
        if self.csm_op is not None:
            acquisition_model = acquisition_model @ self.csm_op

        operator = acquisition_model.gram
        (right_hand_side,) = acquisition_model.H(kdata.data)

        if not torch.all(self.regularization_weight == 0):  # Has Regularization
            operator = operator + self.regularization_weight * self.regularization_op.gram
            right_hand_side = (
                right_hand_side + self.regularization_weight * self.regularization_op.H(self.regularization_data)[0]
            )

        if self.dcf_op is not None:
            # The DCF is used to obtain a good starting point for the CG algorithm.
            # This is equivalten to running the CG algorithm with H = A^H DCF A and b = A^H DCF y
            # for a single iteration.
            (u,) = (acquisition_model.H @ self.dcf_op)(kdata.data)
            (v,) = (acquisition_model.H @ self.dcf_op @ acquisition_model)(u)
            u_flat = u.flatten(start_dim=-3)
            v_flat = v.flatten(start_dim=-3)
            initial_value = (
                unsqueeze_right(torch.linalg.vecdot(u_flat, u_flat) / torch.linalg.vecdot(v_flat, u_flat), 3) * u
            )
        else:
            # The right and side is not a good starting point without DCF.
            initial_value = torch.zeros_like(right_hand_side)
        (img_tensor,) = cg(
            operator,
            right_hand_side,
            initial_value=initial_value,
            max_iterations=self.n_iterations,
            tolerance=0,  # run for a fixed number of iterations
        )
        img = IData.from_tensor_and_kheader(img_tensor, kdata.header)
        return img
