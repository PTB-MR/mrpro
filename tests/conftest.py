import tempfile

import ismrmrd
import pytest
import torch
from ismrmrd import xsd
from mrpro.data import AcqInfo
from mrpro.data import KHeader
from mrpro.data.enums import AcqFlags
from xsdata.models.datatype import XmlDate
from xsdata.models.datatype import XmlTime

from tests import RandomGenerator
from tests.data import Dicom2DTestImage
from tests.phantoms._EllipsePhantomTestData import EllipsePhantomTestData


def generate_random_encodingcounter_properties(generator: RandomGenerator):
    return {
        'kspace_encode_step_1': generator.uint16(),
        'kspace_encode_step_2': generator.uint16(),
        'average': generator.uint16(),
        'slice': generator.uint16(),
        'contrast': generator.uint16(),
        'phase': generator.uint16(),
        'repetition': generator.uint16(),
        'set': generator.uint16(),
        'segment': generator.uint16(),
        'user': generator.uint16_tuple(8),
    }


def generate_random_acquisition_properties(generator: RandomGenerator):
    idx_properties = generate_random_encodingcounter_properties(generator)
    return {
        'flags': generator.uint64(high=2**38 - 1),
        'measurement_uid': generator.uint32(),
        'scan_counter': generator.uint32(),
        'acquisition_time_stamp': generator.uint32(),
        'physiology_time_stamp': generator.uint32_tuple(3),
        'available_channels': generator.uint16(),
        'channel_mask': generator.uint32_tuple(16),
        'discard_pre': generator.uint16(),
        'discard_post': generator.uint16(),
        'center_sample': generator.uint16(),
        'encoding_space_ref': generator.uint16(),
        'sample_time_us': generator.float32(),
        'position': generator.float32_tuple(3),
        'read_dir': generator.float32_tuple(3),
        'phase_dir': generator.float32_tuple(3),
        'slice_dir': generator.float32_tuple(3),
        'patient_table_position': generator.float32_tuple(3),
        'idx': ismrmrd.EncodingCounters(**idx_properties),
        'user_int': generator.uint32_tuple(8),
        'user_float': generator.float32_tuple(8),
    }


def generate_random_trajectory(generator: RandomGenerator, shape=(256, 2)):
    return generator.float32_tensor(shape)


def generate_random_data(generator: RandomGenerator, shape=(32, 256)):
    return generator.complex64_tensor(shape)


@pytest.fixture(scope='session')
def ellipse_phantom():
    return EllipsePhantomTestData()


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


@pytest.fixture(params=({'seed': 0, 'n_coils': 32, 'n_samples': 256},))
def random_acquisition(request):
    seed, n_coils, n_samples = (
        request.param['seed'],
        request.param['n_coils'],
        request.param['n_samples'],
    )
    generator = RandomGenerator(seed)
    kdata = generate_random_data(generator, (n_coils, n_samples))
    trajectory = generate_random_trajectory(generator, (n_samples, 2))
    header = generate_random_acquisition_properties(generator)
    header['flags'] &= ~AcqFlags.ACQ_IS_NOISE_MEASUREMENT.value
    return ismrmrd.Acquisition.from_array(kdata, trajectory, **header)


@pytest.fixture(params=({'seed': 1, 'n_coils': 32, 'n_samples': 256},))
def random_noise_acquisition(request):
    seed, n_coils, n_samples = (
        request.param['seed'],
        request.param['n_coils'],
        request.param['n_samples'],
    )
    generator = RandomGenerator(seed)
    kdata = generate_random_data(generator, (n_coils, n_samples))
    trajectory = generate_random_trajectory(generator, (n_samples, 2))
    header = generate_random_acquisition_properties(generator)
    header['flags'] |= AcqFlags.ACQ_IS_NOISE_MEASUREMENT.value
    return ismrmrd.Acquisition.from_array(kdata, trajectory, **header)


@pytest.fixture(params=({'seed': 0},))
def random_full_ismrmrd_header(request) -> xsd.ismrmrdschema.ismrmrdHeader:
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
        echoTrainLength=generator.uint8(),
        parallelImaging=xsd.parallelImagingType(
            accelerationFactor=xsd.accelerationFactorType(generator.uint8(), generator.uint8()),
            calibrationMode=xsd.calibrationModeType('other'),
        ),
        encodingLimits=xsd.encodingLimitsType(
            kspace_encoding_step_1=xsd.limitType(0, 0, 0),
            kspace_encoding_step_2=xsd.limitType(0, 0, 0),
        ),
    )
    measurement_information = xsd.measurementInformationType(
        measurementID=generator.ascii(10),
        seriesDate=XmlDate(generator.uint16(1970, 2030), generator.uint8(1, 12), generator.uint8(0, 30)),
        seriesTime=XmlTime(generator.uint8(0, 23), generator.uint8(0, 59), generator.uint8(0, 59)),
        sequenceName=generator.ascii(10),
    )

    acquisition_system_information = xsd.acquisitionSystemInformationType(
        systemFieldStrength_T=generator.float32(0, 12),
        systemVendor=generator.ascii(10),
        systemModel=generator.ascii(10),
        receiverChannels=generator.uint16(1, 32),
    )

    sequence_parameters = xsd.sequenceParametersType(
        TR=[generator.float32()],
        TE=[generator.float32()],
        flipAngle_deg=[generator.float32(low=10, high=90)],
        echo_spacing=[generator.float32()],
        sequence_type=generator.ascii(10),
    )

    # TODO: add everything that to the header
    return xsd.ismrmrdschema.ismrmrdHeader(
        encoding=[encoding],
        sequenceParameters=sequence_parameters,
        version=generator.int16(),
        experimentalConditions=xsd.experimentalConditionsType(H1resonanceFrequency_Hz=generator.int32()),
        measurementInformation=measurement_information,
        acquisitionSystemInformation=acquisition_system_information,
    )


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
def random_ismrmrd_file(random_acquisition, random_noise_acquisition, full_header):
    with tempfile.NamedTemporaryFile(suffix='.h5') as file:
        dataset = ismrmrd.Dataset(file.name)
        dataset.append_acquisition(random_acquisition)
        dataset.append_acquisition(random_noise_acquisition)
        dataset.write_xml_header(full_header.toXML())
        dataset.close()

        yield file.name


@pytest.fixture()
def random_acq_info(random_acquisition):
    """Random (not necessarily valid) AcqInfo."""
    acq_info = AcqInfo.from_ismrmrd_acquisitions([random_acquisition])
    return acq_info


@pytest.fixture(params=({'seed': 0},))
def random_kheader(request, random_full_ismrmrd_header, random_acq_info):
    """Random (not necessarily valid) KHeader."""
    seed = request.param['seed']
    generator = RandomGenerator(seed)
    trajectory = generate_random_trajectory(generator)
    kheader = KHeader.from_ismrmrd(
        random_full_ismrmrd_header,
        acq_info=random_acq_info,
        defaults={'trajectory': trajectory},
    )
    return kheader


@pytest.fixture(params=({'seed': 0, 'n_other': 10, 'n_k2': 40, 'n_k1': 20},))
def random_kheader_shape(request, random_acquisition, random_full_ismrmrd_header):
    """Random (not necessarily valid) KHeader with defined shape."""
    # Get dimensions
    seed, n_other, n_k2, n_k1 = (
        request.param['seed'],
        request.param['n_other'],
        request.param['n_k2'],
        request.param['n_k1'],
    )
    generator = RandomGenerator(seed)

    # Generate acquisitions
    random_acq_info = AcqInfo.from_ismrmrd_acquisitions([random_acquisition for _ in range(n_k1 * n_k2 * n_other)])
    n_k0 = int(random_acq_info.number_of_samples[0])
    n_coils = int(random_acq_info.active_channels[0])

    # Generate trajectory
    ktraj = [generate_random_trajectory(generator, shape=(n_k0, 2)) for _ in range(n_k1 * n_k2 * n_other)]

    # Put it all together to a KHeader object
    kheader = KHeader.from_ismrmrd(random_full_ismrmrd_header, acq_info=random_acq_info, defaults={'trajectory': ktraj})
    return kheader, n_other, n_coils, n_k2, n_k1, n_k0


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


def create_parameter_tensor_tuples(parameter_shape=(10, 5, 100, 100, 100), number_of_tensors=2):
    """Create tuples of tensors as input to operators."""
    random_generator = RandomGenerator(seed=0)
    parameter_tensors = random_generator.float32_tensor(size=(number_of_tensors, *parameter_shape), low=1e-10)
    return torch.unbind(parameter_tensors)


COMMON_MR_TRAJECTORIES = pytest.mark.parametrize(
    ('im_shape', 'k_shape', 'nkx', 'nky', 'nkz', 'sx', 'sy', 'sz', 's0', 's1', 's2'),
    [
        # (0) 2d cart mri with 1 coil, no oversampling
        (
            (1, 1, 1, 96, 128),  # img shape
            (1, 1, 1, 96, 128),  # k shape
            (1, 1, 1, 128),  # kx
            (1, 1, 96, 1),  # ky
            (1, 1, 1, 1),  # kz
            'uf',  # kx is uniform
            'uf',  # ky is uniform
            'z',  # zero so no Fourier transform is performed along that dimension
            'uf',  # k0 is uniform
            'uf',  # k1 is uniform
            'z',  # k2 is singleton
        ),
        # (1) 2d cart mri with 1 coil, with oversampling
        (
            (1, 1, 1, 96, 128),
            (1, 1, 1, 128, 192),
            (1, 1, 1, 192),
            (1, 1, 128, 1),
            (1, 1, 1, 1),
            'uf',
            'uf',
            'z',
            'uf',
            'uf',
            'z',
        ),
        # (2) 2d non-Cartesian mri with 2 coils
        (
            (1, 2, 1, 96, 128),
            (1, 2, 1, 16, 192),
            (1, 1, 16, 192),
            (1, 1, 16, 192),
            (1, 1, 1, 1),
            'nuf',  # kx is non-uniform
            'nuf',
            'z',
            'nuf',
            'nuf',
            'z',
        ),
        # (3) 2d cart mri with irregular sampling
        (
            (1, 1, 1, 96, 128),
            (1, 1, 1, 1, 192),
            (1, 1, 1, 192),
            (1, 1, 1, 192),
            (1, 1, 1, 1),
            'uf',
            'uf',
            'z',
            'uf',
            'z',
            'z',
        ),
        # (4) 2d single shot spiral
        (
            (1, 2, 1, 96, 128),
            (1, 1, 1, 1, 192),
            (1, 1, 1, 192),
            (1, 1, 1, 192),
            (1, 1, 1, 1),
            'nuf',
            'nuf',
            'z',
            'nuf',
            'z',
            'z',
        ),
        # (5) 3d nuFFT mri, 4 coils, 2 other
        (
            (2, 4, 16, 32, 64),
            (2, 4, 16, 32, 64),
            (2, 16, 32, 64),
            (2, 16, 32, 64),
            (2, 16, 32, 64),
            'nuf',
            'nuf',
            'nuf',
            'nuf',
            'nuf',
            'nuf',
        ),
        # (6) 2d nuFFT cine mri with 8 cardiac phases, 5 coils
        (
            (8, 5, 1, 64, 64),
            (8, 5, 1, 18, 128),
            (8, 1, 18, 128),
            (8, 1, 18, 128),
            (8, 1, 1, 1),
            'nuf',
            'nuf',
            'z',
            'nuf',
            'nuf',
            'z',
        ),
        # (7) 2d cart cine mri with 9 cardiac phases, 6 coils
        (
            (9, 6, 1, 96, 128),
            (9, 6, 1, 128, 192),
            (9, 1, 1, 192),
            (9, 1, 128, 1),
            (9, 1, 1, 1),
            'uf',
            'uf',
            'z',
            'uf',
            'uf',
            'z',
        ),
        # (8) radial phase encoding (RPE), 8 coils, with oversampling in both FFT and nuFFT directions
        (
            (2, 8, 64, 32, 48),
            (2, 8, 8, 64, 96),
            (2, 1, 1, 96),
            (2, 8, 64, 1),
            (2, 8, 64, 1),
            'uf',
            'nuf',
            'nuf',
            'uf',
            'nuf',
            'nuf',
        ),
        # (9) radial phase encoding (RPE) , 8 coils with non-Cartesian sampling along readout
        (
            (2, 8, 64, 32, 48),
            (2, 8, 8, 64, 96),
            (2, 1, 1, 96),
            (2, 8, 64, 1),
            (2, 8, 64, 1),
            'nuf',
            'nuf',
            'nuf',
            'nuf',
            'nuf',
            'nuf',
        ),
        # (10) stack of stars, 5 other, 3 coil, oversampling in both FFT and nuFFT directions
        (
            (5, 3, 48, 16, 32),
            (5, 3, 96, 18, 64),
            (5, 1, 18, 64),
            (5, 1, 18, 64),
            (5, 96, 1, 1),
            'nuf',
            'nuf',
            'uf',
            'nuf',
            'nuf',
            'uf',
        ),
    ],
)

# Shape combinations for signal models
SHAPE_VARIATIONS_SIGNAL_MODELS = pytest.mark.parametrize(
    ('parameter_shape', 'contrast_dim_shape', 'signal_shape'),
    [
        ((1, 1, 10, 20, 30), (5,), (5, 1, 1, 10, 20, 30)),  # single map with different inversion times
        ((1, 1, 10, 20, 30), (5, 1), (5, 1, 1, 10, 20, 30)),
        ((4, 1, 1, 10, 20, 30), (5, 1), (5, 4, 1, 1, 10, 20, 30)),  # multiple maps along additional batch dimension
        ((4, 1, 1, 10, 20, 30), (5,), (5, 4, 1, 1, 10, 20, 30)),
        ((4, 1, 1, 10, 20, 30), (5, 4), (5, 4, 1, 1, 10, 20, 30)),
        ((3, 1, 10, 20, 30), (5,), (5, 3, 1, 10, 20, 30)),  # multiple maps along other dimension
        ((3, 1, 10, 20, 30), (5, 1), (5, 3, 1, 10, 20, 30)),
        ((3, 1, 10, 20, 30), (5, 3), (5, 3, 1, 10, 20, 30)),
        ((4, 3, 1, 10, 20, 30), (5,), (5, 4, 3, 1, 10, 20, 30)),  # multiple maps along other and batch dimension
        ((4, 3, 1, 10, 20, 30), (5, 4), (5, 4, 3, 1, 10, 20, 30)),
        ((4, 3, 1, 10, 20, 30), (5, 4, 1), (5, 4, 3, 1, 10, 20, 30)),
        ((4, 3, 1, 10, 20, 30), (5, 1, 3), (5, 4, 3, 1, 10, 20, 30)),
        ((4, 3, 1, 10, 20, 30), (5, 4, 3), (5, 4, 3, 1, 10, 20, 30)),
        ((1,), (5,), (5, 1)),  # single voxel
        ((4, 3, 1), (5, 4, 3), (5, 4, 3, 1)),
    ],
)
