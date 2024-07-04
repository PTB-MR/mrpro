"""Direct Reconstruction by Adjoint Fourier Transform."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
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

from typing import Self

from mrpro.algorithms.prewhiten_kspace import prewhiten_kspace
from mrpro.algorithms.reconstruction.Reconstruction import Reconstruction
from mrpro.data._kdata.KData import KData
from mrpro.data.CsmData import CsmData
from mrpro.data.DcfData import DcfData
from mrpro.data.IData import IData
from mrpro.data.KNoise import KNoise
from mrpro.operators.FourierOp import FourierOp
from mrpro.operators.LinearOperator import LinearOperator


class DirectReconstruction(Reconstruction):
    """Direct Reconstruction by Adjoint Fourier Transform."""

    def __init__(
        self,
        fourier_op: LinearOperator,
        csm: None | CsmData = None,
        noise: None | KNoise = None,
        dcf: DcfData | None = None,
    ):
        """Initialize DirectReconstruction.

        Parameters
        ----------
        fourier_op
            Instance of the FourierOperator which adjoint is used for reconstruction.
        csm
            Sensitivity maps for coil combination. If None, no coil combination will be performed.
        noise
            Used for prewhitening
        dcf
            Density compensation. If None, no dcf will be performed.
            Also set to None, if the FourierOperator is already density compensated.
        """
        super().__init__()
        self.fourier_op = fourier_op
        # TODO: Make this buffers once DataBufferMixin is merged
        self.dcf = dcf
        self.csm = csm
        self.noise = noise

    @classmethod
    def from_kdata(cls, kdata: KData, noise: KNoise | None = None, coil_combine: bool = True) -> Self:
        """Create a DirectReconstruction from kdata with default settings.

        Parameters
        ----------
        kdata
            KData to use for trajektory and header information.
        noise
            KNoise used for prewhitening. If None, no prewhitening is performed.
        coil_combine
            if True (default), uses kdata to estimate sensitivity maps
            and perform adaptive coil combine reconstruction
            in the reconstruction.
        """
        if noise is not None:
            kdata = prewhiten_kspace(kdata, noise)
        dcf = DcfData.from_traj_voronoi(kdata.traj)
        fourier_op = FourierOp.from_kdata(kdata)
        self = cls(fourier_op, None, noise, dcf)
        if coil_combine:
            # kdata is prewhitened
            self.recalculate_csm_walsh(kdata, noise=False)
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
        return self.pseudo_inverse(kdata)
