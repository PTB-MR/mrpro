"""Test fixtures for data consistency tests."""

import pytest
from mrpro.data.KData import KData
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.data.traj_calculators.KTrajectoryCartesian import KTrajectoryCartesian
from mrpro.operators.FourierOp import FourierOp
from mrpro.phantoms.EllipsePhantom import EllipsePhantom
from mrpro.utils import RandomGenerator


@pytest.fixture
def kdata():
    matrix = SpatialDimension(x=128, y=128, z=1)
    kdata = EllipsePhantom().kdata(KTrajectoryCartesian.fullysampled(matrix), matrix)
    return kdata


@pytest.fixture
def kdata_noisy(kdata: KData):
    kdata_noisy = kdata.clone()
    kdata_noisy.data += 0.1 * RandomGenerator(123).randn_like(kdata_noisy.data)
    return kdata_noisy


@pytest.fixture
def kdata_us(kdata: KData):
    return kdata[..., ::2, :].clone()


@pytest.fixture
def image_noisy(kdata_noisy: KData):
    fourier_op = FourierOp.from_kdata(kdata_noisy)
    return fourier_op.adjoint(kdata_noisy.data)[0]


@pytest.fixture
def image(kdata: KData):
    fourier_op = FourierOp.from_kdata(kdata)
    return fourier_op.adjoint(kdata.data)[0]


@pytest.fixture
def image_us(kdata_us: KData):
    fourier_op = FourierOp.from_kdata(kdata_us)
    return fourier_op.adjoint(kdata_us.data)[0]
