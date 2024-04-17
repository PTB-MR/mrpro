from __future__ import annotations

from typing import Literal
from typing import Self

from mrpro.algorithms import prewhiten_kspace
from mrpro.algorithms.reconstruction import Reconstruction
from mrpro.data._CsmData import CsmData
from mrpro.data._DcfData import DcfData
from mrpro.data._IData import IData
from mrpro.data._kdata._KData import KData
from mrpro.data._KNoise import KNoise
from mrpro.operators._FourierOp import FourierOp
from mrpro.operators._LinearOperator import LinearOperator


class DirectReconstruction(Reconstruction):
    """Direct Reconstruction by Adjoint Fourier Transform."""

    dcf: DcfData | None
    """Density Compensation Data."""

    csm: CsmData | None
    """Coil Sensitivity Data."""

    noise: KNoise | None
    """Noise Data used for prewhitening."""

    fourier_op: LinearOperator
    """Fourier Operator used for the adjoint."""

    def __init__(
        self,
        fourier_operator: LinearOperator,
        csm: None | CsmData = None,
        noise: None | KNoise = None,
        dcf: DcfData | None = None,
    ):
        """Initialize DirectReconstruction.

        Parameters
        ----------
        fourier_operator
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
        self.fourier_op = fourier_operator
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

    def recalculate_fourierop(self, kdata: KData):
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
            KData used for adjoint reconstruction, which is then used for
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
        adjoint = type(self)(self.fourier_op, dcf=self.dcf, noise=noise)
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
