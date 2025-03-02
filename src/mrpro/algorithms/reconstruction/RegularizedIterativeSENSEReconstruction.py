"""Regularized Iterative SENSE Reconstruction."""

from __future__ import annotations

from collections.abc import Callable

import torch

from mrpro.algorithms.optimizers.cg import cg
from mrpro.algorithms.prewhiten_kspace import prewhiten_kspace
from mrpro.algorithms.reconstruction.DirectReconstruction import DirectReconstruction
from mrpro.data.CsmData import CsmData
from mrpro.data.DcfData import DcfData
from mrpro.data.IData import IData
from mrpro.data.KData import KData
from mrpro.data.KNoise import KNoise
from mrpro.operators.IdentityOp import IdentityOp
from mrpro.operators.LinearOperator import LinearOperator


class RegularizedIterativeSENSEReconstruction(DirectReconstruction):
    r"""Regularized iterative SENSE reconstruction.

    This algorithm solves the problem :math:`\min_{x} \frac{1}{2}||W^\frac{1}{2} (Ax - y)||_2^2 +
    \frac{1}{2}\lambda||Bx - x_0||_2^2`
    by using a conjugate gradient algorithm to solve
    :math:`H x = b` with :math:`H = A^H W A + \lambda B^H B` and :math:`b = A^H W y + \lambda B^H x_0` where :math:`A`
    is the acquisition model (coil sensitivity maps, Fourier operator, k-space sampling), :math:`y` is the acquired
    k-space data, :math:`W` describes the density compensation, :math:`\lambda` is the strength of the regularization
    and :math:`x_0` is the regularization image (i.e. the prior). :math:`B` is a linear operator applied to :math:`x`.
    """

    n_iterations: int
    """Number of CG iterations."""

    regularization_data: torch.Tensor
    """Regularization data (i.e. prior) :math:`x_0`."""

    regularization_weight: torch.Tensor
    r"""Strength of the regularization :math:`\lambda`."""

    regularization_op: LinearOperator
    """Linear operator :math:`B` applied to the current estimate in the regularization term."""

    def __init__(
        self,
        kdata: KData | None = None,
        fourier_op: LinearOperator | None = None,
        csm: Callable | CsmData | None = CsmData.from_idata_walsh,
        noise: KNoise | None = None,
        dcf: DcfData | None = None,
        *,
        n_iterations: int = 5,
        regularization_data: float | torch.Tensor = 0.0,
        regularization_weight: float | torch.Tensor,
        regularization_op: LinearOperator | None = None,
    ) -> None:
        r"""Initialize RegularizedIterativeSENSEReconstruction.

        For a unregularized version of the iterative SENSE algorithm the regularization_weight can be set to ``0``
        or `~mrpro.algorithms.reconstruction.IterativeSENSEReconstruction` algorithm can be used.

        Parameters
        ----------
        kdata
            If `kdata` is provided and `fourier_op` or `dcf` are `None`, then `fourier_op` and `dcf` are estimated
            based on `kdata`. Otherwise `fourier_op` and `dcf` are used as provided.
        fourier_op
            Instance of the `~mrpro.operators.FourierOp` used for reconstruction.
            If `None`, set up based on `kdata`.
        csm
            Sensitivity maps for coil combination. If `None`, no coil combination is carried out, i.e. images for each
            coil are returned. If a `Callable` is provided, coil images are reconstructed using the adjoint of the
            `~mrpro.operators.FourierOp` (including density compensation) and then sensitivity maps are calculated
            using the `Callable`. For this, `kdata` needs also to be provided.
            For examples have a look at the `mrpro.data.CsmData` class e.g. `~mrpro.data.CsmData.from_idata_walsh`
            or `~mrpro.data.CsmData.from_idata_inati`.
        noise
            Noise used for prewhitening. If `None`, no prewhitening is performed
        dcf
            K-space sampling density compensation. If `None`, set up based on `kdata`.
        n_iterations
            Number of CG iterations
        regularization_data
            Regularization data, e.g. a reference image (:math:`x_0`).
        regularization_weight
            Strength of the regularization (:math:`\lambda`).
        regularization_op
            Linear operator :math:`B` applied to the current estimate in the regularization term. If None, nothing is
            applied to the current estimate.

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

        # Create the normal operator as H = A^H W A if the DCF is not None otherwise as H = A^H A.
        # The acquisition model is A = F S if the CSM S is defined otherwise A = F with the Fourier operator F
        csm_op = self.csm.as_operator() if self.csm is not None else IdentityOp()
        precondition_op = self.dcf.as_operator() if self.dcf is not None else IdentityOp()
        operator = (self.fourier_op @ csm_op).H @ precondition_op @ (self.fourier_op @ csm_op)

        # Calculate the right-hand-side as b = A^H W y if the DCF is not None otherwise as b = A^H y.
        (right_hand_side,) = (self.fourier_op @ csm_op).H(precondition_op(kdata.data)[0])

        # Add regularization
        if not torch.all(self.regularization_weight == 0):
            operator = operator + self.regularization_weight * self.regularization_op.H @ self.regularization_op
            right_hand_side += self.regularization_weight * self.regularization_op.H(self.regularization_data)[0]

        img_tensor = cg(
            operator,
            right_hand_side,
            initial_value=right_hand_side,
            max_iterations=self.n_iterations,
            tolerance=0.0,
        )
        img = IData.from_tensor_and_kheader(img_tensor, kdata.header)
        return img
