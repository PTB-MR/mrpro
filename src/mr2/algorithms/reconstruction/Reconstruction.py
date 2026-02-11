"""Reconstruction module."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal

import torch
from typing_extensions import Self

from mr2.algorithms.prewhiten_kspace import prewhiten_kspace
from mr2.data.CsmData import CsmData
from mr2.data.DcfData import DcfData
from mr2.data.IData import IData
from mr2.data.KData import KData
from mr2.data.KNoise import KNoise
from mr2.operators.FourierOp import FourierOp
from mr2.operators.LinearOperator import LinearOperator


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
            k-space data to determine trajectory and recon/encoding matrix from.
        """
        self.fourier_op = FourierOp.from_kdata(kdata)
        self.dcf = DcfData.from_traj_voronoi(kdata.traj)
        return self

    def recalculate_csm(
        self,
        kdata: KData,
        csm_calculation: Callable[[IData], CsmData] = CsmData.from_idata_walsh,
        noise: KNoise | None | Literal[False] = None,
    ) -> Self:
        """Update (in place) the CSM from KData.

        Performs a direct reconstruction without coil combination
        and estimates coil sensitivity maps from the result.

        Parameters
        ----------
        kdata
            k-space data used for adjoint reconstruction (including DCF-weighting if available), which is then used for
            CSM estimation.
        csm_calculation
            Function to calculate csm expecting idata as input and returning csmdata. For examples have a look at the
            `~mr2.data.CsmData`.
        noise
            Noise measurement for prewhitening.
            If `None`, `self.noise` (if previously set) is used.
            If `False`, no prewhitening is performed even if `self.noise` is set.
            Use this if the `kdata` is already prewhitened.
        """
        image = self.direct_reconstruction(kdata, csm=False, noise=noise)
        self.csm = csm_calculation(image)
        return self

    def direct_reconstruction(
        self,
        kdata: KData,
        *,
        csm: CsmData | None | Literal[False] = None,
        noise: KNoise | None | Literal[False] = None,
    ) -> IData:
        """Direct reconstruction of the MR acquisition.

        Here we use :math:`S^H F^H W` to calculate the image data using
        the coil sensitivity operator :math:`S`,
        the Fourier operator :math:`F`,
        and the density compensation operator :math:`W`.

        Parameters
        ----------
        kdata
            k-space data
        csm
            Coil sensitivity maps used for coil combination.
            If `None`, ``self.csm`` is used.
            If `False`, no coil combination is performed.
        noise
            Noise measurement for prewhitening.
            If `None`, ``self.noise`` is used.
            If `False`, no prewhitening is performed.

        Returns
        -------
            image data
        """
        noise_data = self.noise if noise is None else (None if noise is False else noise)
        csm_data = self.csm if csm is None else (None if csm is False else csm)

        if noise_data is not None:
            kdata = prewhiten_kspace(kdata, noise_data)
        operator = self.fourier_op
        if csm_data is not None:
            operator = operator @ csm_data.as_operator()
        if self.dcf is not None:
            operator = self.dcf.as_operator() @ operator
        (img_tensor,) = operator.H(kdata.data)
        img = IData.from_tensor_and_kheader(img_tensor, kdata.header)
        return img
