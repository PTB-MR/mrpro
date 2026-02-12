"""PyTest fixtures for reconstruction tests."""

from collections.abc import Callable

import pytest
from mr2.algorithms.reconstruction import DirectReconstruction
from mr2.data import CsmData, DcfData, KData
from mr2.data.traj_calculators import KTrajectoryCartesian
from mr2.operators import FourierOp
from tests.conftest import ismrmrd_cart as _ismrmrd_cart  # noqa: F401
from tests.data import IsmrmrdRawTestData


@pytest.fixture(scope='session')
def cartesian_kdata(ismrmrd_cart: IsmrmrdRawTestData) -> KData:
    """Create Cartesian KData from shared ISMRMRD test fixture."""
    return KData.from_file(ismrmrd_cart.filename, KTrajectoryCartesian())


@pytest.fixture
def explicit_components() -> Callable[[KData], tuple[FourierOp, CsmData, DcfData]]:
    """Create explicit FourierOp, CSM, and DCF from KData."""

    def _explicit_components(kdata: KData) -> tuple[FourierOp, CsmData, DcfData]:
        direct_reconstruction = DirectReconstruction(kdata=kdata, csm=None)
        csm = CsmData.from_idata_walsh(direct_reconstruction(kdata))
        dcf = DcfData.from_traj_voronoi(kdata.traj)
        return FourierOp.from_kdata(kdata), csm, dcf

    return _explicit_components
