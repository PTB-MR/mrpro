"""Iterative SENSE reconstruction."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Literal
from typing import Self

from mrpro.algorithms.prewhiten_kspace import prewhiten_kspace
from mrpro.algorithms.optimizers import cg
from mrpro.algorithms.reconstruction import DirectReconstruction
from mrpro.algorithms.reconstruction import Reconstruction
from mrpro.data.CsmData import CsmData
from mrpro.data.DcfData import DcfData
from mrpro.data.IData import IData
from mrpro.data._kdata.KData import KData
from mrpro.data.KNoise import KNoise
from mrpro.operators.FourierOp import FourierOp
from mrpro.operators.LinearOperator import LinearOperator


class IterativeSenseReconstruction(Reconstruction):
    """Iterative SENSE reconstruction.

    This algorithm minizes the problem

    min_x 0.5||W^0.5 (Ax - y)||_2^2

    by using a conjugate gradient algorithm to solve

    H x = b

    with H = A^H W A and b = A^H W y

    where A is the acquisition model (coil sensitivity maps, Fourier operator, k-space sampling), y is the acquired
    k-space data and W describes the density compensation.

    More information can be found here:
    Pruessmann, K. P., Weiger, M., Boernert, P. & Boesiger, P. Advances in sensitivity encoding with arbitrary k-space
    trajectories. Magn. Reson. Imaging 46, 638-651 (2001). https://doi.org/10.1002/mrm.1241

    """
    def __init__(
        self,
        fourier_op: LinearOperator,
        n_max_iter: int,
        csm: CsmData | None = None,
        noise: None | KNoise = None,
        dcf: DcfData | None = None,
    ) -> None:
        """Initialize DirectReconstruction.

        Parameters
        ----------
        fourier_operator
            Instance of the FourierOperator which adjoint is used for reconstruction.
        csm
            Sensitivity maps for coil combination
        n_max_iter
            Maximum number of CG iterations
        noise
            Used for prewhitening
        dcf
            Density compensation. If None, no dcf will be performed.
            Also set to None, if the FourierOperator is already density compensated.
        """
        super().__init__()
        self.fourier_op = fourier_op
        self.n_max_iter = n_max_iter
        # TODO: Make this buffers once DataBufferMixin is merged
        self.csm = csm
        self.noise = noise
        self.dcf = dcf



    @classmethod
    def from_kdata(cls, kdata: KData, n_max_iter: int, noise: KNoise | None = None, coil_combine: bool = True) -> Self:
        """Create a IterativeSenseReconstruction from kdata with default settings.

        Parameters
        ----------
        kdata
            KData to use for trajektory and header information
        n_max_iter
            Maximum number of CG iterations
        noise
            KNoise used for prewhitening. If None, no prewhitening is performed
        """
        if noise is not None:
            kdata = prewhiten_kspace(kdata, noise)
        dcf = DcfData.from_traj_voronoi(kdata.traj)
        fourier_op = FourierOp.from_kdata(kdata)
        adjoint = DirectReconstruction(fourier_op, dcf=dcf, noise=noise)
        image = adjoint(kdata)
        csm = CsmData.from_idata_walsh(image)
        return cls(fourier_op, n_max_iter, csm, noise, dcf)

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

        operator = self.fourier_op.H
        if self.dcf is not None:
            operator = operator @ self.dcf.as_operator()
        if self.csm is not None:
            operator = self.csm.as_operator() @ operator
        (right_hand_side,) = operator.to(device)(kdata.data)
        if self.csm is not None:
            operator = operator @ self.csm.as_operator()
        operator = operator @ self.fourier_op
        operator = operator.to(device)
        img_tensor = cg(operator, right_hand_side, initial_value=right_hand_side, max_iterations=self.n_max_iter)
        img = IData.from_tensor_and_kheader(img_tensor, kdata.header)
        return img
