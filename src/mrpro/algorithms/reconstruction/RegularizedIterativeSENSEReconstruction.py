"""Regularized Iterative SENSE Reconstruction by adjoint Fourier transform."""

from __future__ import annotations

from collections.abc import Callable

import torch

from mrpro.algorithms.optimizers.cg import cg
from mrpro.algorithms.prewhiten_kspace import prewhiten_kspace
from mrpro.algorithms.reconstruction.DirectReconstruction import DirectReconstruction
from mrpro.data._kdata.KData import KData
from mrpro.data.CsmData import CsmData
from mrpro.data.DcfData import DcfData
from mrpro.data.IData import IData
from mrpro.data.KNoise import KNoise
from mrpro.operators.IdentityOp import IdentityOp
from mrpro.operators.LinearOperator import LinearOperator


class RegularizedIterativeSENSEReconstruction(DirectReconstruction):
    r"""Regularized iterative SENSE reconstruction.

    This algorithm solves the problem :math:`min_x \frac{1}{2}||(Ax - y)||_2^2 +
    \frac{1}{2} \lambda ||Bx - x_r||_2^2`
    by using a conjugate gradient algorithm to solve
    :math:`H x = b` with :math:`H = A^H A + \lambda B` and :math:`b = A^H y + \lambda x_r` where :math:`A`
    is the acquisition model (coil sensitivity maps, Fourier operator, k-space sampling), :math:`y` is the acquired
    k-space data, :math:`\lambda` is the regularization weight.
    :math:`x_0` is the regularization data (i.e. the prior) and :math:`B` is a linear operator applied to :math:`x`
    in the regularization term.
    """

    n_iterations: int
    """Number of CG iterations."""

    regularization_data: torch.Tensor
    """Regularization data (i.e. prior) :math:`x_r`."""

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
        """Initialize RegularizedIterativeSENSEReconstruction.

        For a unregularized version of the iterative SENSE algorithm the regularization_weight can be set to 0 or
        IterativeSENSEReconstruction algorithm can be used.

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
            K-space sampling density compensation.
            Only used to obtain the starting point of the iterative reconstruction.
        n_iterations
            Number of CG iterations
        regularization_data
            Regularization data, e.g. a reference image (:math:`x_0`).
        regularization_weight
            Strength of the regularization (:math:`L`).
        regularization_op
            Linear operator :math:`B` applied to the current estimate in the regularization term. If None, nothing is
            applied to the current estimate.


        Raises
        ------
        ValueError
            If the kdata and fourier_op are None or if csm is a Callable but kdata is None.
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

        # Create the the acquisition model A = Fourier CSM # and normal operator H = A^H A
        csm_op = self.csm.as_operator() if self.csm is not None else IdentityOp()
        acquisition_model = self.fourier_op @ csm_op
        operator = acquisition_model.gram

        # Calculate the right-hand-side as b = A^H y
        (right_hand_side,) = (acquisition_model.H)(kdata.data)

        # Add regularization
        if not torch.all(self.regularization_weight == 0):
            operator = operator + (self.regularization_weight * self.regularization_op)
            right_hand_side += self.regularization_weight * self.regularization_data

        # if available, use dcf for initial value
        if self.dcf is not None:
            (initial_value,) = (acquisition_model.H @ self.dcf.as_operator())(kdata.data)
        else:
            initial_value = right_hand_side

        img_tensor = cg(
            operator,
            right_hand_side,
            initial_value=initial_value,
            max_iterations=self.n_iterations,
            tolerance=0.0,
        )
        img = IData.from_tensor_and_kheader(img_tensor, kdata.header)
        return img
