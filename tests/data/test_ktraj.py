# from pathlib import Path

from mrpro.data import KData
from mrpro.data._KTrajectory import RadialKTraj2D


def test_KTrajectory_calc(random_ismrmrd_file):
    k = KData.from_file(random_ismrmrd_file, RadialKTraj2D(golden_angle=True))
    assert k is not None

# fpath = Path('/data/brahma01/Datasets/MRpro_data/meas_MID00020_FID02083_FLASH.mrd')
# a = KData.from_file(fpath, RadialKTraj2D(golden_angle=True)) # , multi_slice=False
# b = 1
