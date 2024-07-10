"""PyTest fixtures for the data tests."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pytest
import torch
from ismrmrd import xsd
from mrpro.data import AcqInfo

from tests import RandomGenerator
from tests.data import Dicom2DTestImage
from tests.conftest import generate_random_data


@pytest.fixture(params=({'seed': 0},))
def cartesian_grid(request):
    generator = RandomGenerator(request.param['seed'])

    def generate(n_k2: int, n_k1: int, n_k0: int, jitter: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k0_range = torch.arange(n_k0)
        k1_range = torch.arange(n_k1)
        k2_range = torch.arange(n_k2)
        ky, kz, kx = torch.meshgrid(k1_range, k2_range, k0_range, indexing='xy')
        if jitter > 0:
            kx = kx + generator.float32_tensor((n_k2, n_k1, n_k0), high=jitter)
            ky = ky + generator.float32_tensor((n_k2, n_k1, n_k0), high=jitter)
            kz = kz + generator.float32_tensor((n_k2, n_k1, n_k0), high=jitter)
        return kz.unsqueeze(0), ky.unsqueeze(0), kx.unsqueeze(0)

    return generate

@pytest.fixture(params=({'seed': 0},))
def random_mandatory_ismrmrd_header(request) -> xsd.ismrmrdschema.ismrmrdHeader:
    """Generate a full header, i.e. all values used in
    KHeader.from_ismrmrd_header() are set."""

    seed = request.param['seed']
    generator = RandomGenerator(seed)
    encoding = xsd.encodingType(
        trajectory=xsd.trajectoryType('other'),
        encodedSpace=xsd.encodingSpaceType(
            matrixSize=xsd.matrixSizeType(x=generator.int16(), y=generator.uint8(), z=generator.uint8()),
            fieldOfView_mm=xsd.fieldOfViewMm(x=generator.uint8(), y=generator.uint8(), z=generator.uint8()),
        ),
        reconSpace=xsd.encodingSpaceType(
            matrixSize=xsd.matrixSizeType(x=generator.uint8(), y=generator.uint8(), z=generator.uint8()),
            fieldOfView_mm=xsd.fieldOfViewMm(x=generator.uint8(), y=generator.uint8(), z=generator.uint8()),
        ),
        encodingLimits=xsd.encodingLimitsType(),
    )
    experimental_conditions = xsd.experimentalConditionsType(H1resonanceFrequency_Hz=generator.int32())
    return xsd.ismrmrdschema.ismrmrdHeader(encoding=[encoding], experimentalConditions=experimental_conditions)


@pytest.fixture()
def random_acq_info(random_acquisition):
    """Random (not necessarily valid) AcqInfo."""
    acq_info = AcqInfo.from_ismrmrd_acquisitions([random_acquisition])
    return acq_info


@pytest.fixture(params=({'seed': 0, 'n_other': 2, 'n_coils': 16, 'n_z': 32, 'n_y': 128, 'n_x': 256},))
def random_test_data(request):
    seed, n_other, n_coils, n_z, n_y, n_x = (
        request.param['seed'],
        request.param['n_other'],
        request.param['n_coils'],
        request.param['n_z'],
        request.param['n_y'],
        request.param['n_x'],
    )
    generator = RandomGenerator(seed)
    test_data = generate_random_data(generator, (n_other, n_coils, n_z, n_y, n_x))
    return test_data


@pytest.fixture(scope='session')
def dcm_2d(ellipse_phantom, tmp_path_factory):
    """Single 2D dicom image."""
    dcm_filename = tmp_path_factory.mktemp('mrpro') / 'dicom_2d.dcm'
    dcm_idata = Dicom2DTestImage(filename=dcm_filename, phantom=ellipse_phantom.phantom)
    return dcm_idata


@pytest.fixture(scope='session', params=({'n_images': 7},))
def dcm_multi_echo_times(request, ellipse_phantom, tmp_path_factory):
    """Multiple 2D dicom images with different echo times."""
    n_images = request.param['n_images']
    path = tmp_path_factory.mktemp('mrpro_multi_dcm')
    te = 2.0
    dcm_image_data = []
    for _ in range(n_images):
        dcm_filename = path / f'dicom_te_{int(te)}.dcm'
        dcm_image_data.append(Dicom2DTestImage(filename=dcm_filename, phantom=ellipse_phantom.phantom, te=te))
        te += 1.0
    return dcm_image_data


@pytest.fixture(scope='session', params=({'n_images': 7},))
def dcm_multi_echo_times_multi_folders(request, ellipse_phantom, tmp_path_factory):
    """Multiple 2D dicom images with different echo times each saved in a different folder."""
    n_images = request.param['n_images']
    te = 2.0
    dcm_image_data = []
    for _ in range(n_images):
        path = tmp_path_factory.mktemp(f'mrpro_multi_dcm_te_{int(te)}')
        dcm_filename = path / f'dicom_te_{int(te)}.dcm'
        dcm_image_data.append(Dicom2DTestImage(filename=dcm_filename, phantom=ellipse_phantom.phantom, te=te))
        te += 1.0
    return dcm_image_data
