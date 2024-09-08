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
from mrpro.operators.LinearOperator import LinearOperator


class RegularizedIterativeSENSEReconstruction(DirectReconstruction):
    r"""Regularized iterative SENSE reconstruction.

    This algorithm solves the problem :math:`min_x \frac{1}{2}||W^\frac{1}{2} (Ax - y)||_2^2 +
    \frac{1}{2}L||Bx - x_0||_2^2`
    by using a conjugate gradient algorithm to solve
    :math:`H x = b` with :math:`H = A^H W A + L B` and :math:`b = A^H W y + L x_0` where :math:`A`
    is the acquisition model (coil sensitivity maps, Fourier operator, k-space sampling), :math:`y` is the acquired
    k-space data, :math:`W` describes the density compensation, :math:`L` is the strength of the regularisation and
    :math:`x_0` is the regularisation image (i.e. the prior). :math:`B` is a linear operator applied to :math:`x`.
    """

    n_iterations: int
    """Number of CG iterations."""

    regularisation_data: torch.Tensor
    """Regularisation data (i.e. prior) :math:`x_0`."""

    regularisation_weight: torch.Tensor
    """Strength of the regularisation :math:`L`."""

    linear_op: LinearOperator | None
    """Linear operator :math:`B` applied to the current estimate in the regularisation term."""

    def __init__(
        self,
        kdata: KData | None = None,
        fourier_op: LinearOperator | None = None,
        csm: Callable | CsmData | None = CsmData.from_idata_walsh,
        noise: KNoise | None = None,
        dcf: DcfData | None = None,
        *,
        n_iterations: int = 5,
        regularisation_data: float | torch.Tensor = 0.0,
        regularisation_weight: float | torch.Tensor,
        linear_op: LinearOperator | None = None,
    ) -> None:
        """Initialize RegularizedIterativeSENSEReconstruction.

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
            K-space sampling density compensation. If None, set up based on kdata.
        n_iterations
            Number of CG iterations
        regularisation_data
            Regularisation data, e.g. a reference image (:math:`x_0`).
        regularisation_weight
            Strength of the regularisation (:math:`L`).
        linear_op
            Linear operator :math:`B` applied to the current estimate in the regularisation term. If None, nothing is
            applied to the current estimate.


        Raises
        ------
        ValueError
            If the kdata and fourier_op are None or if csm is a Callable but kdata is None.
        """
        super().__init__(kdata, fourier_op, csm, noise, dcf)
        self.n_iterations = n_iterations
        self.regularisation_data = torch.as_tensor(regularisation_data)
        self.regularisation_weight = torch.as_tensor(regularisation_weight)
        self.linear_op = linear_op

    def _self_adjoint_operator(self) -> LinearOperator:
        """Create the self-adjoint operator.

        Create the acquisition model as :math:`A = F S` if the CSM :math:`S` is defined otherwise :math:`A = F` with
        the Fourier operator :math:`F`.

        Create the self-adjoint operator as :math:`H = A^H W A` if the DCF is not None otherwise as :math:`H = A^H A`.
        """
        operator = self.fourier_op @ self.csm.as_operator() if self.csm is not None else self.fourier_op

        if self.dcf is not None:
            dcf_operator = self.dcf.as_operator()
            # Create H = A^H W A
            operator = operator.H @ dcf_operator @ operator
        else:
            # Create H = A^H A
            operator = operator.H @ operator

        return operator

    def _right_hand_side(self, kdata: KData) -> torch.Tensor:
        """Calculate the right-hand-side of the normal equation.

        Create the acquisition model as :math:`A = F S` if the CSM :math:`S` is defined otherwise :math:`A = F` with
        the Fourier operator :math:`F`.

        Calculate the right-hand-side as :math:`b = A^H W y` if the DCF is not None otherwise as :math:`b = A^H y`.

        Parameters
        ----------
        kdata
            k-space data to reconstruct.
        """
        device = kdata.data.device
        operator = self.fourier_op @ self.csm.as_operator() if self.csm is not None else self.fourier_op

        if self.dcf is not None:
            dcf_operator = self.dcf.as_operator()
            # Calculate b = A^H W y
            (right_hand_side,) = operator.to(device).H(dcf_operator(kdata.data)[0])
        else:
            # Calculate b = A^H y
            (right_hand_side,) = operator.to(device).H(kdata.data)

        return right_hand_side

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
        device = kdata.data.device
        if self.noise is not None:
            kdata = prewhiten_kspace(kdata, self.noise.to(device))

        operator = self._self_adjoint_operator().to(device)
        right_hand_side = self._right_hand_side(kdata)

        # Add regularisation
        if not torch.all(self.regularisation_weight == 0):
            if self.linear_op is None:
                operator = operator + self.regularisation_weight * self.linear_op
            else:
                operator = operator + self.regularisation_weight
            right_hand_side += self.regularisation_weight * self.regularisation_data

        img_tensor = cg(
            operator, right_hand_side, initial_value=right_hand_side, max_iterations=self.n_iterations, tolerance=0.0
        )
        img = IData.from_tensor_and_kheader(img_tensor, kdata.header)
        return img