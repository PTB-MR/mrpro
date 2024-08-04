"""Iterative SENSE Reconstruction by adjoint Fourier transform."""

from __future__ import annotations

from typing import Self

import torch

from mrpro.algorithms.optimizers.cg import cg
from mrpro.algorithms.prewhiten_kspace import prewhiten_kspace
from mrpro.algorithms.reconstruction.Reconstruction import Reconstruction
from mrpro.data._kdata.KData import KData
from mrpro.data.CsmData import CsmData
from mrpro.data.DcfData import DcfData
from mrpro.data.IData import IData
from mrpro.data.KNoise import KNoise
from mrpro.operators.FourierOp import FourierOp
from mrpro.operators.LinearOperator import LinearOperator


class IterativeSENSEReconstruction(Reconstruction):
    r"""Iterative SENSE reconstruction.

    This algorithm solves the problem :math:`min_x \frac{1}{2}||W^\frac{1}{2} (Ax - y)||_2^2`
    by using a conjugate gradient algorithm to solve
    :math:`H x = b` with :math:`H = A^H W A` and :math:`b = A^H W y` where :math:`A` is the acquisition model
    (coil sensitivity maps, Fourier operator, k-space sampling), :math:`y` is the acquired k-space data and :math:`W`
    describes the density compensation [PRU2001]_ .

    Note: In [PRU2001]_ a k-space filter is applied as a final step to null all k-space values outside the k-space
    coverage. This is not done here.

    .. [PRU2001] Pruessmann K, Weiger M, Boernert P, and Boesiger P (2001), Advances in sensitivity encoding with
       arbitrary k-space trajectories. MRI 46, 638-651. https://doi.org/10.1002/mrm.1241

    """

    n_iterations: int
    """Number of CG iterations."""

    def __init__(
        self,
        fourier_op: LinearOperator,
        n_iterations: int,
        csm: CsmData | None = None,
        noise: KNoise | None = None,
        dcf: DcfData | None = None,
    ) -> None:
        """Initialize IterativeSENSEReconstruction.

        Parameters
        ----------
        fourier_op
            Instance of the FourierOperator used for reconstruction
        n_iterations
            Number of CG iterations
        csm
            Sensitivity maps for coil combination
        noise
            Used for prewhitening
        dcf
            Density compensation. If None, no dcf will be performed.
            Also set to None, if the FourierOperator is already density compensated.
        """
        super().__init__()
        self.fourier_op = fourier_op
        self.n_iterations = n_iterations
        # TODO: Make this buffers once DataBufferMixin is merged
        self.csm = csm
        self.noise = noise
        self.dcf = dcf

    @classmethod
    def from_kdata(
        cls,
        kdata: KData,
        noise: KNoise | None = None,
        csm: CsmData | None = None,
        *,
        n_iterations: int = 10,
    ) -> Self:
        """Create a IterativeSENSEReconstruction from kdata with default settings.

        Parameters
        ----------
        kdata
            KData to use for trajectory and header information
        noise
            KNoise used for prewhitening. If None, no prewhitening is performed
        csm
            Sensitivity maps. If None, no CSM operator will be applied.
        n_iterations
            Number of CG iterations
        """
        if noise is not None:
            kdata = prewhiten_kspace(kdata, noise)
        dcf = DcfData.from_traj_voronoi(kdata.traj)
        fourier_op = FourierOp.from_kdata(kdata)
        return cls(fourier_op, n_iterations, csm, noise, dcf)

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

        img_tensor = cg(
            operator, right_hand_side, initial_value=right_hand_side, max_iterations=self.n_iterations, tolerance=0.0
        )
        img = IData.from_tensor_and_kheader(img_tensor, kdata.header)
        return img
