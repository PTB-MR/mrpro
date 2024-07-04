"""Reconstruction module."""

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

from abc import ABC
from abc import abstractmethod
from typing import Literal
from typing import Self

import torch

from mrpro.algorithms.prewhiten_kspace import prewhiten_kspace
from mrpro.data._kdata.KData import KData
from mrpro.data.CsmData import CsmData
from mrpro.data.DcfData import DcfData
from mrpro.data.IData import IData
from mrpro.data.KNoise import KNoise
from mrpro.operators.FourierOp import FourierOp
from mrpro.operators.LinearOperator import LinearOperator


class Reconstruction(torch.nn.Module, ABC):
    """A Reconstruction."""

    dcf: DcfData | None
    """Density Compensation Data."""

    csm: CsmData | None
    """Coil Sensitivity Data."""

    noise: KNoise | None
    """Noise Data used for prewhitening."""

    fourier_op: LinearOperator
    """Fourier Operator."""

    @abstractmethod
    def forward(self, kdata: KData) -> IData:
        """Apply the reconstruction."""

    # Required for type hinting
    def __call__(self, kdata: KData) -> IData:
        """Apply the reconstruction."""
        return super().__call__(kdata)

    def recalculate_fourierop(self, kdata: KData) -> Self:
        """Update (in place) the Fourier Operator, e.g. for a new trajectory.

        Also recalculates the DCF.

        Parameters
        ----------
        kdata
            KData to determine trajectory and recon/encoding matrix from.
        """
        self.fourier_op = FourierOp.from_kdata(kdata)
        self.dcf = DcfData.from_traj_voronoi(kdata.traj)
        return self

    def recalculate_csm_walsh(self, kdata: KData, noise: KNoise | None | Literal[False] = None) -> Self:
        """Update (in place) the CSM from KData using Walsh.

        Parameters
        ----------
        kdata
            KData used for pseudo-inverse reconstruction, which is then used for
            Walsh CSM estimation.
        noise
            Noise measurement for prewhitening.
            If None, self.noise (if previously set) is used.
            If False, no prewithening is performed even if self.noise is set.
            Use this if the kdata is already prewhitened.
        """
        if noise is False:
            noise = None
        elif noise is None:
            noise = self.noise
        recon = type(self)(self.fourier_op, dcf=self.dcf, noise=noise)
        image = recon.pseudo_inverse(kdata)
        self.csm = CsmData.from_idata_walsh(image)
        return self

    def pseudo_inverse(self, kdata: KData) -> IData:
        """Pseudo-inverse of the MR acquisition.

        Here we use S^H F^H W to calculate the image data using the coil sensitivity operator S, the Fourier operator F
        and the density compensation operator W. S and W are optional.

        Parameters
        ----------
        kdata
            k-space data

        Returns
        -------
            image data
        """
        device = kdata.data.device
        if self.noise is not None:
            kdata = prewhiten_kspace(kdata, self.noise.to(device))
        operator = self.fourier_op
        if self.csm is not None:
            operator = operator @ self.csm.as_operator()
        if self.dcf is not None:
            operator = self.dcf.as_operator() @ operator
        operator = operator.to(device)
        (img_tensor,) = operator.H(kdata.data)
        img = IData.from_tensor_and_kheader(img_tensor, kdata.header)
        return img
