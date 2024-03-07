"""Tests for the KData class."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import pytest
import torch
from mrpro.data import KData
from mrpro.data import KTrajectory
from mrpro.data.traj_calculators._KTrajectoryCalculator import DummyTrajectory
from mrpro.operators import FastFourierOp

from tests.data import IsmrmrdRawTestData
from tests.helper import relative_image_difference


@pytest.fixture(scope='session')
def ismrmrd_cart(ellipse_phantom, tmp_path_factory):
    """Fully sampled cartesian data set."""
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_cart.h5'
    ismrmrd_kdata = IsmrmrdRawTestData(
        filename=ismrmrd_filename,
        noise_level=0.0,
        repetitions=3,
        phantom=ellipse_phantom.phantom,
    )
    return ismrmrd_kdata


@pytest.fixture(scope='session')
def ismrmrd_cart_invalid_reps(tmp_path_factory):
    """Fully sampled cartesian data set."""
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_cart.h5'
    ismrmrd_kdata = IsmrmrdRawTestData(
        filename=ismrmrd_filename,
        noise_level=0.0,
        repetitions=3,
        flag_invalid_reps=True,
    )
    return ismrmrd_kdata


@pytest.fixture(scope='session')
def ismrmrd_cart_random_us(ellipse_phantom, tmp_path_factory):
    """Randomly undersampled cartesian data set with repetitions."""
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_cart.h5'
    ismrmrd_kdata = IsmrmrdRawTestData(
        filename=ismrmrd_filename,
        noise_level=0.0,
        repetitions=3,
        acceleration=4,
        sampling_order='random',
        phantom=ellipse_phantom.phantom,
    )
    return ismrmrd_kdata


def test_KData_from_file(ismrmrd_cart):
    """Read in data from file."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    assert kdata is not None


def test_KData_random_cart_undersampling(ismrmrd_cart_random_us):
    """Read data with different random Cartesian undersampling in multiple
    repetitions."""
    kdata = KData.from_file(ismrmrd_cart_random_us.filename, DummyTrajectory())
    assert kdata is not None


def test_KData_random_cart_undersampling_shape(ismrmrd_cart_random_us):
    """Check shape of KData with random Cartesian undersampling."""
    kdata = KData.from_file(ismrmrd_cart_random_us.filename, DummyTrajectory())
    # check if the number of repetitions is correct
    assert kdata.data.shape[-5] == ismrmrd_cart_random_us.repetitions
    # check if the number of phase encoding lines per repetition is correct
    assert kdata.data.shape[-2] == ismrmrd_cart_random_us.matrix_size // ismrmrd_cart_random_us.acceleration


def test_KData_raise_wrong_trajectory_shape(ismrmrd_cart):
    """Wrong KTrajectory shape raises exception."""
    kx = ky = kz = torch.zeros(1, 2, 3, 4)
    trajectory = KTrajectory(kz, ky, kx, repeat_detection_tolerance=None)
    with pytest.raises(ValueError):
        _ = KData.from_file(ismrmrd_cart.filename, trajectory)


def test_KData_from_file_diff_nky_for_rep(ismrmrd_cart_invalid_reps):
    """Multiple repetitions with different number of phase encoding lines is
    not supported."""
    with pytest.raises(ValueError, match=r'Number of \((k2 k1\)) points in '):
        KData.from_file(ismrmrd_cart_invalid_reps.filename, DummyTrajectory())


def test_KData_kspace(ismrmrd_cart):
    """Read in data and verify k-space by comparing reconstructed image."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    ff_op = FastFourierOp(dim=(-1, -2))
    (reconstructed_img,) = ff_op.adjoint(kdata.data)

    # Due to discretisation artifacts the reconstructed image will be different to the reference image. Using standard
    # testing functions such as numpy.testing.assert_almost_equal fails because there are few voxels with high
    # differences along the edges of the elliptic objects.
    assert relative_image_difference(reconstructed_img[0, 0, 0, ...], ismrmrd_cart.imref) <= 0.05


@pytest.mark.parametrize(('field', 'value'), [('b0', 11.3), ('tr', [24.3])])
def test_KData_modify_header(ismrmrd_cart, field, value):
    """Overwrite some parameters in the header."""
    parameter_dict = {field: value}
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory(), header_overwrites=parameter_dict)
    assert getattr(kdata.header, field) == value


def test_KData_to_complex128(ismrmrd_cart):
    """Change KData dtype complex128."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_complex128 = kdata.to(dtype=torch.complex128)
    assert kdata_complex128.data.dtype == torch.complex128
    assert kdata_complex128.traj.kx.dtype == torch.float64
    assert kdata_complex128.traj.ky.dtype == torch.float64
    assert kdata_complex128.traj.kz.dtype == torch.float64


@pytest.mark.cuda()
def test_KData_to_cuda(ismrmrd_cart):
    """Test KData.to to move data to CUDA memory."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    cuda_device = f'cuda:{torch.cuda.current_device()}'
    kdata_cuda = kdata.to(device=cuda_device)
    assert kdata_cuda.data.is_cuda
    assert kdata_cuda.traj.kz.is_cuda
    assert kdata_cuda.traj.ky.is_cuda
    assert kdata_cuda.traj.kx.is_cuda


@pytest.mark.cuda()
def test_KData_cuda(ismrmrd_cart):
    """Move KData object to CUDA memory."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_cuda = kdata.cuda()
    assert kdata_cuda.data.is_cuda
    assert kdata_cuda.traj.kz.is_cuda
    assert kdata_cuda.traj.ky.is_cuda
    assert kdata_cuda.traj.kx.is_cuda


@pytest.mark.cuda()
def test_KData_cpu(ismrmrd_cart):
    """Move KData object to CUDA memory and back to CPU memory."""
    kdata = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    kdata_cpu = kdata.cuda().cpu()
    assert kdata_cpu.data.is_cpu
    assert kdata_cpu.traj.kz.is_cpu
    assert kdata_cpu.traj.ky.is_cpu
    assert kdata_cpu.traj.kx.is_cpu
