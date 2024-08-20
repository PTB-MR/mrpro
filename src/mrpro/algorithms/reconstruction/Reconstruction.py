"""Reconstruction module."""

from abc import ABC, abstractmethod
from typing import Literal, Self

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
            KData used for adjoint reconstruction (including DCF-weighting if available), which is then used for
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
        image = recon.direct_reconstruction(kdata)
        self.csm = CsmData.from_idata_walsh(image)
        return self

    def direct_reconstruction(self, kdata: KData) -> IData:
        """Direct reconstruction of the MR acquisition.

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
