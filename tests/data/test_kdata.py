from mrpro.data import KData
from mrpro.data._KTrajectory import DummyTrajectory


def test_KData_from_randomfile(random_ismrmrd_file):
    k = KData.from_file(random_ismrmrd_file, DummyTrajectory())
    assert k is not None
