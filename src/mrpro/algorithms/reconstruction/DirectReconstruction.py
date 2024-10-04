"""Direct Reconstruction by Adjoint Fourier Transform."""

from collections.abc import Callable

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
        kdata: KData | None = None,
        fourier_op: LinearOperator | None = None,
        csm: Callable[[IData], CsmData] | CsmData | None = CsmData.from_idata_walsh,
        noise: KNoise | None = None,
        dcf: DcfData | None = None,
    ):
        """Initialize DirectReconstruction.

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
            K-space sampling density compensation. If None, set up based on kdata.

        Raises
        ------
        ValueError
            If the kdata and fourier_op are None or if csm is a Callable but kdata is None.
        """
        super().__init__()
        if fourier_op is None:
            if kdata is None:
                raise ValueError('Either kdata or fourier_op needs to be defined.')
            self.fourier_op = FourierOp.from_kdata(kdata)
        else:
            self.fourier_op = fourier_op

        if kdata is not None and dcf is None:
            self.dcf = DcfData.from_traj_voronoi(kdata.traj)
        else:
            self.dcf = dcf

        self.noise = noise

        if csm is None or isinstance(csm, CsmData):
            self.csm = csm
        else:
            if kdata is None:
                raise ValueError('kdata needs to be defined to calculate the sensitivity maps.')
            self.recalculate_csm(kdata, csm)

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
        return self.direct_reconstruction(kdata)
