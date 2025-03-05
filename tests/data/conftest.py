"""PyTest fixtures for the data tests."""

from collections.abc import Callable

import pytest
import torch
from ismrmrd import xsd
from mrpro.data import Rotation, SpatialDimension
from mrpro.utils import unsqueeze_left

from tests import RandomGenerator
from tests.data import DicomTestImage, IsmrmrdRawTestData


@pytest.fixture(params=({'seed': 0},))
def cartesian_grid(request) -> Callable[[int, int, int, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Cartesian grid generator.

    Generates a 3D cartesian grid with optional jitter.
    Shape of the returned tensors is `(1, 1, n_k2, n_k1, n_k0)`.
    """
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
        kz, ky, kx = (unsqueeze_left(x, 2) for x in (kz, ky, kx))
        return kz, ky, kx

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


@pytest.fixture(params=({'seed': 0, 'n_other': 2, 'n_coils': 8, 'n_z': 16, 'n_y': 32, 'n_x': 64},))
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
    test_data = generator.complex64_tensor((n_other, n_coils, n_z, n_y, n_x))
    return test_data


@pytest.fixture(scope='session')
def dcm_2d(ellipse_phantom, tmp_path_factory):
    """Single 2D dicom image."""
    dcm_filename = tmp_path_factory.mktemp('mrpro_2d') / 'dicom.dcm'
    return (DicomTestImage(filename=dcm_filename, phantom=ellipse_phantom.phantom),)


@pytest.fixture(scope='session')
def dcm_2d_multi_echo_times(ellipse_phantom, tmp_path_factory):
    """Multiple 2D dicom images with different echo times."""
    n_echoes = 7
    path = tmp_path_factory.mktemp('mrpro_2d_multi_echo')
    te = 0.02
    dcm_image_data = []
    series_instance_uid = None
    for idx in range(n_echoes):
        dcm_filename = path / f'dicom_{idx}.dcm'
        dcm_image_data.append(
            DicomTestImage(
                filename=dcm_filename, phantom=ellipse_phantom.phantom, te=te, series_instance_uid=series_instance_uid
            )
        )
        te += 0.01
        series_instance_uid = dcm_image_data[-1].series_instance_uid
    return dcm_image_data


@pytest.fixture(scope='session')
def dcm_2d_multi_echo_times_multi_folders(ellipse_phantom, tmp_path_factory):
    """Multiple 2D dicom images with different echo times each saved in a different folder."""
    n_echoes = 7
    te = 0.02
    dcm_image_data = []
    for idx in range(n_echoes):
        path = tmp_path_factory.mktemp(f'mrpro_2d_multi_echo_{idx}')
        dcm_filename = path / 'dicom.dcm'
        dcm_image_data.append(DicomTestImage(filename=dcm_filename, phantom=ellipse_phantom.phantom, te=te))
        te += 0.01
    return dcm_image_data


@pytest.fixture(scope='session')
def dcm_m2d_multi_orientation(ellipse_phantom, tmp_path_factory):
    """Multiple 2D dicom images with different orientation."""
    path = tmp_path_factory.mktemp('mrpro_m2d_multi_ori')
    vec_z = SpatialDimension(1.0, 0, 0)
    vec_y = SpatialDimension(0, 1.0, 0)
    vec_x = SpatialDimension(0, 0, 1.0)
    orientations = [Rotation.from_directions(vec_z, vec_y, vec_x), Rotation.from_directions(vec_y, vec_x, vec_z)]
    dcm_image_data = []
    for idx, slice_orientation in enumerate(orientations):
        dcm_filename = path / f'dicom_{idx}.dcm'
        dcm_image_data.append(
            DicomTestImage(filename=dcm_filename, phantom=ellipse_phantom.phantom, slice_orientation=slice_orientation)
        )
    return dcm_image_data


@pytest.fixture(scope='session')
def dcm_3d(ellipse_phantom, tmp_path_factory):
    """3D dicom image in a single dicom file."""
    path = tmp_path_factory.mktemp('mrpro_3d')
    dcm_filename = path / 'dicom.dcm'
    return (DicomTestImage(filename=dcm_filename, phantom=ellipse_phantom.phantom, slice_offset=[-2.0, 0.0, 2.0, 4.0]),)


@pytest.fixture(scope='session')
def dcm_3d_multi_echo(ellipse_phantom, tmp_path_factory):
    """3D dicom images with different echo times, each in a single dicom file."""
    n_echoes = 7
    te = 0.02
    path = tmp_path_factory.mktemp('mrpro_3d_multi_echo')
    dcm_image_data = []
    series_instance_uid = None
    for idx in range(n_echoes):
        dcm_filename = path / f'dicom_{idx}.dcm'
        dcm_image_data.append(
            DicomTestImage(
                filename=dcm_filename,
                phantom=ellipse_phantom.phantom,
                slice_offset=[-2.0, 0.0, 2.0, 4.0],
                te=te,
                series_instance_uid=series_instance_uid,
            )
        )
        series_instance_uid = dcm_image_data[-1].series_instance_uid
        te += 0.01
    return dcm_image_data


@pytest.fixture(scope='session')
def dcm_3d_multi_echo_multi_cardiac_phases(ellipse_phantom, tmp_path_factory):
    """3D dicom images with different echo times, each in a single dicom file."""
    n_echoes = 7
    n_cardiac_phases = 3

    path = tmp_path_factory.mktemp('mrpro_3d_multi_echo_multi_cardiac_phases')
    dcm_image_data = []
    time_after_rpeak = 0.1
    idx = 0
    series_instance_uid = None
    for _ in range(n_cardiac_phases):
        te = 0.02
        for _ in range(n_echoes):
            dcm_filename = path / f'dicom_{idx}.dcm'
            dcm_image_data.append(
                DicomTestImage(
                    filename=dcm_filename,
                    phantom=ellipse_phantom.phantom,
                    slice_offset=[-2.0, 0.0, 2.0, 4.0],
                    te=te,
                    time_after_rpeak=time_after_rpeak,
                    series_instance_uid=series_instance_uid,
                )
            )
            series_instance_uid = dcm_image_data[-1].series_instance_uid
            te += 0.01
            idx += 1
        time_after_rpeak += 0.1
    return dcm_image_data


@pytest.fixture(scope='session')
def dcm_3d_multi_orientation(ellipse_phantom, tmp_path_factory):
    """Multiple 3D dicom images with different orientation."""
    path = tmp_path_factory.mktemp('mrpro_3d_multi_ori')
    vec_z = SpatialDimension(1.0, 0, 0)
    vec_y = SpatialDimension(0, 1.0, 0)
    vec_x = SpatialDimension(0, 0, 1.0)
    orientations = [Rotation.from_directions(vec_z, vec_y, vec_x), Rotation.from_directions(vec_y, vec_x, vec_z)]
    dcm_image_data = []
    for idx, slice_orientation in enumerate(orientations):
        dcm_filename = path / f'dicom_{idx}.dcm'
        dcm_image_data.append(
            DicomTestImage(
                filename=dcm_filename,
                phantom=ellipse_phantom.phantom,
                slice_orientation=slice_orientation,
                slice_offset=[-2.0, 0.0, 2.0, 4.0],
            )
        )
    return dcm_image_data


@pytest.fixture(scope='session')
def ismrmrd_cart_bodycoil_and_surface_coil(ellipse_phantom, tmp_path_factory):
    """Fully sampled cartesian data set with bodycoil and surface coil data."""
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_cart.h5'
    ismrmrd_kdata = IsmrmrdRawTestData(
        filename=ismrmrd_filename,
        noise_level=0.0,
        repetitions=3,
        phantom=ellipse_phantom.phantom,
        add_bodycoil_acquisitions=True,
    )
    return ismrmrd_kdata


@pytest.fixture(scope='session')
def ismrmrd_cart_with_calibration_lines(ellipse_phantom, tmp_path_factory):
    """Undersampled Cartesian data set with calibration lines."""
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_cart.h5'
    ismrmrd_kdata = IsmrmrdRawTestData(
        filename=ismrmrd_filename,
        noise_level=0.0,
        repetitions=1,
        acceleration=2,
        phantom=ellipse_phantom.phantom,
        n_separate_calibration_lines=16,
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


@pytest.fixture(scope='session')
def ismrmrd_rad(ellipse_phantom, tmp_path_factory):
    """Data set with uniform radial k-space sampling."""
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_rad.h5'
    ismrmrd_data = IsmrmrdRawTestData(
        filename=ismrmrd_filename,
        noise_level=0.0,
        repetitions=3,
        phantom=ellipse_phantom.phantom,
        trajectory_type='radial',
        acceleration=4,
    )
    return ismrmrd_data
