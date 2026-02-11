"""PyTest fixtures for reconstruction tests."""

import pytest
from mrpro.data import KData
from mrpro.data.traj_calculators import KTrajectoryCartesian
from tests.conftest import ismrmrd_cart as _ismrmrd_cart  # noqa: F401
from tests.data import IsmrmrdRawTestData


@pytest.fixture(scope='session')
def cartesian_kdata(ismrmrd_cart: IsmrmrdRawTestData) -> KData:
    """Create Cartesian KData from shared ISMRMRD test fixture."""
    return KData.from_file(ismrmrd_cart.filename, KTrajectoryCartesian())
