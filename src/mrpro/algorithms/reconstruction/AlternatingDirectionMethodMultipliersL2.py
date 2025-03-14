"""Total Variation (TV)-Regularized Reconstruction using PDHG."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from mrpro.algorithms.prewhiten_kspace import prewhiten_kspace
from mrpro.algorithms.reconstruction.DirectReconstruction import DirectReconstruction
from mrpro.algorithms.reconstruction.RegularizedIterativeSENSEReconstruction import (
    RegularizedIterativeSENSEReconstruction,
)
from mrpro.data.CsmData import CsmData
from mrpro.data.DcfData import DcfData
from mrpro.data.IData import IData
from mrpro.data.KData import KData
from mrpro.data.KNoise import KNoise
from mrpro.operators import LinearOperatorMatrix, ProximableFunctionalSeparableSum
from mrpro.operators.FiniteDifferenceOp import FiniteDifferenceOp
from mrpro.operators.functionals import L1NormViewAsReal
from mrpro.operators.LinearOperator import LinearOperator


class AlternatingDirectionMethodMultipliersL2(DirectReconstruction):
    r"""Alternating Direction Method of Multipliers for L2 data consistency terms.

    This algorithm solves the problem :math:`min_x \frac{1}{2}||(Ax - y)||_2^2 + \sum_i l_i ||\nabla_i x||_1`
    by using the PDHG-algorithm. :math:`A` is the acquisition model (coil sensitivity maps, Fourier operator,
    k-space sampling), :math:`y` is the acquired k-space data, :math:`l_i` are the strengths of the regularization
    along the different dimensions and :math:`\nabla_i` is the finite difference operator applied to :math:`x` along
    different dimensions :math:`i`.

    This algorithm solves the problem :math:`\min_x f(x) + g(z) \quad \text{subject to} \quad  Ax + Bz = c` with the
    identification of


    $f(x) = \lambda \| x \|_1$, $g(z)= \frac{1}{2}||Ez - y||_2^2$, $A = I$, $B= -\nabla$ and $c = 0$

    then we can define a scaled form of the ADMM algorithm which solves

    $ \mathcal{F}(x) = \frac{1}{2}||Ex - y||_2^2 + \lambda \| \nabla x \|_1 $

    by doing

    $x_{k+1} = \mathrm{argmin}_x \lambda \| x \|_1 + \frac{\rho}{2}||x - \nabla z_k + u_k||_2^2$ (A)

    $z_{k+1} = \mathrm{argmin}_z \frac{1}{2}||Ez - y||_2^2 + \frac{\rho}{2}||x_{k+1} - \nabla z + u_k||_2^2$ (B)

    $u_{k+1} = u_k + x_{k+1} - \nabla z_{k+1}$ (C)

    The first step is the poximal mapping of the L1-norm of x which is a soft-thresholding of $x$:
    $S_{\lambda/\rho}(\nabla z_k - u_k)$. The second step is a regularized iterative SENSE update of $z$ and the final
    step updates the dual variable $u$.
    """

    n_iterations: int
    """Number of ADMM iterations."""

    regularization_weights: torch.Tensor
    """Strengths of the regularization along different dimensions :math:`l_i`."""

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
        n_iterations: int = 32,
        regularization_weights: Sequence[float] | Sequence[torch.Tensor],
        regularization_op: LinearOperator,
    ) -> None:
        """Initialize AlternatingDirectionMethodMultipliersL2.

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
            Number of ADMM iterations
        regularization_weights
            Strengths of the regularization (:math:`l_i`). Each entry is the regularization weight along a dimension of
            the reconstructed image starting at the back. E.g. (1,) will apply TV with l=1 along dimension (-1,).
            (3,0,2) will apply TV with l=2 along dimension (-1) and TV with l=3 along (-3).

        Raises
        ------
        ValueError
            If the kdata and fourier_op are None or if csm is a Callable but kdata is None.
        """
        super().__init__(kdata, fourier_op, csm, noise, dcf)
        self.n_iterations = n_iterations
        self.regularization_weights = torch.as_tensor(regularization_weights)
        self.data_weight = 0.5

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

        # Finite difference operator and corresponding L1-norm
        nabla_operator = [
            (FiniteDifferenceOp(dim=(dim - len(self.regularization_weights),), mode='forward'),)
            for dim, weight in enumerate(self.regularization_weights)
            if weight != 0
        ]
        l1 = [weight * L1NormViewAsReal() for weight in self.regularization_weights if weight != 0]

        tv = LinearOperatorMatrix(nabla_operator)
        f = ProximableFunctionalSeparableSum(*l1)

        # Initial value
        initial_value = acquisition_operator.H(
            self.dcf.as_operator()(kdata.data)[0] if self.dcf is not None else kdata.data
        )[0]

        regularized_iterative_sense = RegularizedIterativeSENSEReconstruction(
            fourier_op=self.fourier_op,
            csm=self.csm,
            n_iterations=10,
            regularization_weight=self.data_weight,
            regularization_op=nabla_operator,
        )

        img_tensor_z = initial_value.clone()
        img_tensor_u = torch.zeros_like(img_tensor_z)
        data_weight = 0.5
        for _ in range(self.n_iterations):
            # Proximal mapping of x (soft-thresholding)
            img_tensor_x = f.prox([im - img_tensor_u for im in tv(img_tensor_z)], 1 / data_weight)[0]

            # Regularized iterative SENSE
            regularized_iterative_sense.regularization_data = img_tensor_x + img_tensor_u
            img_tensor_z = regularized_iterative_sense(kdata).data

            # Update u
            img_tensor_u = img_tensor_u + img_tensor_x - nabla_operator(img_tensor_z)[0]

        img = IData.from_tensor_and_kheader(img_tensor_z, kdata.header)
        return img
