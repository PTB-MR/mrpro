from __future__ import annotations

from mrpro.algorithms.preprocess import prewhiten_kspace
from mrpro.algorithms.reconstruction import Reconstruction
from mrpro.data import CsmData
from mrpro.data import DcfData
from mrpro.data import IData
from mrpro.data import KData
from mrpro.data import KNoise
from mrpro.operators import FourierOp
from mrpro.operators import LinearOperator
from mrpro.operators import SensitivityOp


class AdjointReconstruction(Reconstruction):
    """Adjoint Reconstruction."""

    def __init__(
        self,
        fourier_operator: LinearOperator,
        csm: None | CsmData = None,
        noise: None | KNoise = None,
        dcf: DcfData | None = None,
    ):
        """Initialize AdjointReconstruction.

        Parameters
        ----------
        fourier_operator
            Instance of the FourierOperator which adjoint is used for reconstruction.
        csm
            Sensitivity maps for coil combination. If None, no coil combination will be performed.
        noise
            Used for Prewhitening
        dcf
            Density compensation. If None, no dcf will be performed.
            Also set to None, if the FourierOperator is already density compensated.
        """
        super().__init__()
        self.FourierOperator = fourier_operator
        self.dcf = dcf
        self.csm = csm
        self.noise = noise

    @classmethod
    def from_kdata(cls, kdata: KData, noise: KNoise | None = None, coil_combine: bool = True) -> AdjointReconstruction:
        """Create an AdjointReconstruction from kdata with default settings.

        Parameters
        ----------
        kdata
            KData to use for trajektory and header information
        noise
            KNoise used for whitening
        coil_combine
            if True (default), uses kdata to estimate sensitivity maps
            and perform adaptive coil combine reconstruction
        """
        if noise is not None:
            kdata = prewhiten_kspace(kdata, noise)
        dcf = DcfData.from_traj_voronoi(kdata.traj)
        fourier_op = FourierOp.from_kdata(kdata)
        self = cls(fourier_op, None, noise, dcf)
        if coil_combine:
            self.recalculate_csm_walsh(kdata)
        return self

    def recalculate_fourierop(self, kdata: KData):
        """Update the Fourier Operator, e.g. for a new trajectory.

        Parameters
        ----------
        kdata
            KData to determine trajectory and recon/encoding matrix from.
        """
        self.FourierOperator = FourierOp.from_kdata(kdata)
        self.dcf = DcfData.from_traj_voronoi(kdata.traj)
        return self

    def recalculate_csm_walsh(self, kdata: KData, noise: KNoise | None = None):
        """Update the CSM from KData using Walsh.

        Parameters
        ----------
        kdata
            KData used for adjoint reconstruction, which is then used for
            Walsh CSM estimation.
        noise
            Noise measurement for prewhitening (optional)
        """
        adjoint = AdjointReconstruction(self.FourierOperator, dcf=self.dcf, noise=noise)
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
        image
            the reconstruced image.
        """
        if self.noise is not None:
            kdata = prewhiten_kspace(kdata, self.noise)
        operator = self.FourierOperator
        if self.dcf is not None:
            operator = operator * self.dcf.data
        if self.csm is not None:
            operator = operator @ SensitivityOp(self.csm)
        (image_data,) = operator.H(kdata.data)
        image = IData.from_tensor_and_kheader(image_data, kdata.header)
        return image
