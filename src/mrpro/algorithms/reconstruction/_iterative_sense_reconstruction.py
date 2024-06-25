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

from typing import Self

import torch

from mrpro.algorithms.optimizers import cg
from mrpro.algorithms.reconstruction import DirectReconstruction
from mrpro.algorithms.reconstruction import Reconstruction
from mrpro.data._CsmData import CsmData
from mrpro.data._DcfData import DcfData
from mrpro.data._IData import IData
from mrpro.data._kdata._KData import KData
from mrpro.operators._FourierOp import FourierOp


class IterativeSenseReconstruction(Reconstruction):
    """Iterative SENSE reconstruction.

    This algorithm minizes the problem

    min_x 0.5||W^0.5 (Ax - y)||_2^2

    by using a conjugate gradient algorithm to solve

    H x = b

    with H = A^H W A and b = A^H W y

    where A is the acquisition model (coil sensitivity maps, Fourier operator, k-space sampling), y is the acquired
    k-space data and W describes the density compensation.
    Note: It is assumed that the input `y` is already pre-whitened

    More information can be found here:
    Pruessmann, K. P., Weiger, M., Boernert, P. & Boesiger, P. Advances in sensitivity encoding with arbitrary k-space
    trajectories. Magn. Reson. Imaging 46, 638-651 (2001). https://doi.org/10.1002/mrm.1241

    """

    acquisition_model: FourierOp
    """Acquisition Operator (= A)"""

    initial_val: torch.Tensor | None
    """Initial value (=x )"""

    dcf: DcfData | None
    """Density Compensation Data."""

    n_max_iter: int
    """Maximum number of CG iterations."""

    def __init__(
        self,
        acquisition_model: FourierOp,
        n_max_iter: int,
        initial_val: torch.Tensor | None = None,
        dcf: DcfData | None = None,
    ):
        """Initialize IterativeSenseReconstruction.

        Parameters
        ----------
        acquisition_model
            Instance of the LinearOperator representing the acquisition model.
        n_max_iter
            Maximum number of CG iterations.
        initial_val
            Initial value for the reconstruction (optional).
        dcf
            Density compensation data. If None, no dcf will be performed.
        """
        super().__init__()
        self.acquisition_model = acquisition_model
        self.initial_val = initial_val
        # TODO: Make this buffers once DataBufferMixin is merged
        self.dcf = dcf
        self.n_max_iter = n_max_iter

    @classmethod
    def from_kdata(cls, kdata: KData, n_max_iter: int) -> Self:
        """Create an IterativeSenseReconstruction from KData with default settings.

        Parameters
        ----------
        kdata
            KData containing trajectory and header information.
        n_max_iter
            Maximum number of CG iterations.

        # adjoint = DirectReconstruction(acquisition_model, dcf=dcf)

        # image = adjoint(kdata)
        # csm = CsmData.from_idata_walsh(image)

        Returns
        -------
        IterativeSenseReconstruction
            Instance of IterativeSenseReconstruction initialized with default settings.
        """
        dcf = DcfData.from_traj_voronoi(kdata.traj)
        acquisition_model = FourierOp.from_kdata(kdata)
        return cls(acquisition_model, n_max_iter, initial_val=None, dcf=dcf)

    def recalculate_fourierop(self, kdata: KData) -> Self:
        """Recalculate the Fourier Operator and DCF.

        Parameters
        ----------
        kdata
            KData used to determine trajectory and recon/encoding matrix.

        Returns
        -------
        IterativeSenseReconstruction
            Updated instance with recalculated Fourier Operator and DCF.
        """
        self.acquisition_model = FourierOp.from_kdata(kdata)
        self.dcf = DcfData.from_traj_voronoi(kdata.traj)
        return self

    def recalculate_csm_walsh(self, kdata: KData) -> Self:
        """Recalculate Coil Sensitivity Maps using Walsh method.

        Parameters
        ----------
        kdata
            KData used for adjoint reconstruction.

        Returns
        -------
        IterativeSenseReconstruction
            Updated instance with recalculated Coil Sensitivity Maps.
        """
        adjoint = DirectReconstruction(self.acquisition_model, dcf=self.dcf)
        image = adjoint(kdata)
        self.csm = CsmData.from_idata_walsh(image)
        return self

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
        operator = self.acquisition_model @ self.csm.as_operator()
        if self.dcf is not None:
            operator = self.dcf.as_operator() @ operator
        operator = operator.to(device)
        (right_hand_side,) = operator.H(kdata.data)
        operator = self.csm.as_operator().H @ self.acquisition_model.H @ operator
        img_tensor = cg(operator, right_hand_side, initial_value=right_hand_side, max_iterations=self.n_max_iter)
        img = IData.from_tensor_and_kheader(img_tensor, kdata.header)
        return img
