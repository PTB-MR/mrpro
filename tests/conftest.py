"""PyTest fixtures for the mrpro package."""

import tempfile

import ismrmrd
import pytest
import torch
from ismrmrd import xsd
from mrpro.data import AcqInfo, KHeader, KTrajectory
from mrpro.data.enums import AcqFlags
from xsdata.models.datatype import XmlDate, XmlTime

from tests import RandomGenerator
from tests.data import IsmrmrdRawTestData
from tests.phantoms import EllipsePhantomTestData


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
        'read_dir': (1, 0, 0),  # read, phase and slice have to form rotation
        'phase_dir': (0, 1, 0),
        'slice_dir': (0, 0, 1),
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
    return ismrmrd.Acquisition.from_array(kdata.numpy(), trajectory.numpy(), **header)


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
    return ismrmrd.Acquisition.from_array(kdata.numpy(), trajectory.numpy(), **header)


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


@pytest.fixture
def random_ismrmrd_file(random_acquisition, random_noise_acquisition, full_header):
    with tempfile.NamedTemporaryFile(suffix='.h5') as file:
        dataset = ismrmrd.Dataset(file.name)
        dataset.append_acquisition(random_acquisition)
        dataset.append_acquisition(random_noise_acquisition)
        dataset.write_xml_header(full_header.toXML())
        dataset.close()

        yield file.name


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


@pytest.fixture
def random_acq_info(random_acquisition):
    """Random (not necessarily valid) AcqInfo."""
    acq_info = AcqInfo.from_ismrmrd_acquisitions([random_acquisition])
    return acq_info


@pytest.fixture(params=({'seed': 0, 'n_other': 4, 'n_k2': 12, 'n_k1': 14},))
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


def create_uniform_traj(nk):
    """Create a tensor of uniform points with predefined shape nk."""
    kidx = torch.where(torch.tensor(nk[1:]) > 1)[0]
    if len(kidx) > 1:
        raise ValueError('nk is allowed to have at most one non-singleton dimension')
    if len(kidx) >= 1:
        # kidx+1 because we searched in nk[1:]
        n_kpoints = nk[kidx + 1]
        k = torch.linspace(-n_kpoints // 2, n_kpoints // 2 - 1, n_kpoints, dtype=torch.float32)
        views = [1 if i != n_kpoints else -1 for i in nk]
        k = k.view(*views).expand(list(nk))
    else:
        k = torch.zeros(nk)
    return k


def create_traj(k_shape, nkx, nky, nkz, type_kx, type_ky, type_kz):
    """Create trajectory with random entries."""
    random_generator = RandomGenerator(seed=0)
    k_list = []
    for spacing, nk in zip([type_kz, type_ky, type_kx], [nkz, nky, nkx], strict=True):
        if spacing == 'non-uniform':
            k = random_generator.float32_tensor(size=nk, low=-1, high=1) * max(nk)
        elif spacing == 'uniform':
            k = create_uniform_traj(nk)
        elif spacing == 'zero':
            k = torch.zeros(nk)
        k_list.append(k)
    trajectory = KTrajectory(k_list[0], k_list[1], k_list[2], repeat_detection_tolerance=None)
    return trajectory


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
def ismrmrd_cart_high_res(ellipse_phantom, tmp_path_factory):
    """Fully sampled cartesian data set."""
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_cart_high_res.h5'
    ismrmrd_kdata = IsmrmrdRawTestData(
        filename=ismrmrd_filename,
        matrix_size=256,
        noise_level=0.0,
        repetitions=3,
        phantom=ellipse_phantom.phantom,
    )
    return ismrmrd_kdata


COMMON_MR_TRAJECTORIES = pytest.mark.parametrize(
    ('im_shape', 'k_shape', 'nkx', 'nky', 'nkz', 'type_kx', 'type_ky', 'type_kz', 'type_k0', 'type_k1', 'type_k2'),
    [
        (  # (0) 2d Cartesian single coil, no oversampling
            (1, 1, 1, 96, 128),  # im_shape
            (1, 1, 1, 96, 128),  # k_shape
            (1, 1, 1, 128),  # nkx
            (1, 1, 96, 1),  # nky
            (1, 1, 1, 1),  # nkz
            'uniform',  # type_kx
            'uniform',  # type_ky
            'zero',  # type_kz
            'uniform',  # type_k0
            'uniform',  # type_k1
            'zero',  # type_k2
        ),
        (  # (1) 2d Cartesian single coil, with oversampling
            (1, 1, 1, 96, 128),  # im_shape
            (1, 1, 1, 128, 192),  # k_shape
            (1, 1, 1, 192),  # nkx
            (1, 1, 128, 1),  # nky
            (1, 1, 1, 1),  # nkz
            'uniform',  # type_kx
            'uniform',  # type_ky
            'zero',  # type_kz
            'uniform',  # type_k0
            'uniform',  # type_k1
            'zero',  # type_k2
        ),
        (  # (2) 2d non-Cartesian mri with 2 coils
            (1, 2, 1, 96, 128),  # im_shape
            (1, 2, 1, 16, 192),  # k_shape
            (1, 1, 16, 192),  # nkx
            (1, 1, 16, 192),  # nky
            (1, 1, 1, 1),  # nkz
            'non-uniform',  # type_kx
            'non-uniform',  # type_ky
            'zero',  # type_kz
            'non-uniform',  # type_k0
            'non-uniform',  # type_k1
            'zero',  # type_k2
        ),
        (  # (3) 2d Cartesian with irregular sampling
            (1, 1, 1, 96, 128),  # im_shape
            (1, 1, 1, 1, 192),  # k_shape
            (1, 1, 1, 192),  # nkx
            (1, 1, 1, 192),  # nky
            (1, 1, 1, 1),  # nkz
            'uniform',  # type_kx
            'uniform',  # type_ky
            'zero',  # type_kz
            'uniform',  # type_k0
            'zero',  # type_k1
            'zero',  # type_k2
        ),
        (  # (4) 2d single shot spiral
            (1, 2, 1, 96, 128),  # im_shape
            (1, 2, 1, 1, 192),  # k_shape
            (1, 1, 1, 192),  # nkx
            (1, 1, 1, 192),  # nky
            (1, 1, 1, 1),  # nkz
            'non-uniform',  # type_kx
            'non-uniform',  # type_ky
            'zero',  # type_kz
            'non-uniform',  # type_k0
            'zero',  # type_k1
            'zero',  # type_k2
        ),
        (  # (5) 3d single shot spiral, 4 coils, 2 other
            (2, 4, 16, 32, 64),  # im_shape
            (2, 4, 1, 64, 1),  # k_shape
            (2, 1, 64, 1),  # nkx
            (2, 1, 64, 1),  # nky
            (2, 1, 64, 1),  # nkz
            'non-uniform',  # type_kx
            'non-uniform',  # type_ky
            'non-uniform',  # type_kz
            'zero',  # type_k0
            'non-uniform',  # type_k1
            'zero',  # type_k2
        ),
        (  # (6) 3d non-uniform, 3 coils, 2 other
            (2, 3, 10, 12, 14),  # im_shape
            (2, 3, 6, 8, 10),  # k_shape
            (2, 6, 8, 10),  # nkx
            (2, 6, 8, 10),  # nky
            (2, 6, 8, 10),  # nkz
            'non-uniform',  # type_kx
            'non-uniform',  # type_ky
            'non-uniform',  # type_kz
            'non-uniform',  # type_k0
            'non-uniform',  # type_k1
            'non-uniform',  # type_k2
        ),
        (  # (7) 2d non-uniform cine with 2 cardiac phases, 3 coils, same traj for each phase
            (2, 3, 1, 64, 64),  # im_shape
            (2, 3, 1, 18, 128),  # k_shape
            (1, 1, 18, 128),  # nkx
            (1, 1, 18, 128),  # nky
            (1, 1, 1, 1),  # nkz
            'non-uniform',  # type_kx
            'non-uniform',  # type_ky
            'zero',  # type_kz
            'non-uniform',  # type_k0
            'non-uniform',  # type_k1
            'zero',  # type_k2
        ),
        (  # (8) 2d non-uniform cine with 2 cardiac phases, 3 coils
            (2, 3, 1, 64, 64),  # im_shape
            (2, 3, 1, 18, 128),  # k_shape
            (2, 1, 18, 128),  # nkx
            (2, 1, 18, 128),  # nky
            (2, 1, 1, 1),  # nkz
            'non-uniform',  # type_kx
            'non-uniform',  # type_ky
            'zero',  # type_kz
            'non-uniform',  # type_k0
            'non-uniform',  # type_k1
            'zero',  # type_k2
        ),
        (  # (9) 2d cartesian cine with 2 cardiac phases, 3 coils
            (2, 3, 1, 96, 128),  # im_shape
            (2, 3, 1, 128, 192),  # k_shape
            (2, 1, 1, 192),  # nkx
            (2, 1, 128, 1),  # nky
            (2, 1, 1, 1),  # nkz
            'uniform',  # type_kx
            'uniform',  # type_ky
            'zero',  # type_kz
            'uniform',  # type_k0
            'uniform',  # type_k1
            'zero',  # type_k2
        ),
        (  # (10) radial phase encoding (RPE), 3 coils, with oversampling in both FFT and non-uniform directions
            (2, 3, 6, 8, 10),  # im_shape
            (2, 3, 8, 10, 12),  # k_shape
            (2, 1, 1, 12),  # nkx
            (2, 8, 10, 1),  # nky
            (2, 8, 10, 1),  # nkz
            'uniform',  # type_kx
            'non-uniform',  # type_ky
            'non-uniform',  # type_kz
            'uniform',  # type_k0
            'non-uniform',  # type_k1
            'non-uniform',  # type_k2
        ),
        (  # (11) radial phase encoding (RPE), 2 coils with non-Cartesian sampling along readout
            (2, 2, 6, 8, 10),  # im_shape
            (2, 2, 8, 10, 12),  # k_shape
            (2, 1, 1, 12),  # nkx
            (2, 8, 10, 1),  # nky
            (2, 8, 10, 1),  # nkz
            'non-uniform',  # type_kx
            'non-uniform',  # type_ky
            'non-uniform',  # type_kz
            'non-uniform',  # type_k0
            'non-uniform',  # type_k1
            'non-uniform',  # type_k2
        ),
        (  # (12) stack of stars, 3 other, 2 coil, oversampling in both FFT and non-uniform directions
            (3, 2, 10, 8, 8),  # im_shape
            (3, 2, 12, 8, 10),  # k_shape
            (3, 1, 8, 10),  # nkx
            (3, 1, 8, 10),  # nky
            (3, 12, 1, 1),  # nkz
            'non-uniform',  # type_kx
            'non-uniform',  # type_ky
            'uniform',  # type_kz
            'non-uniform',  # type_k0
            'non-uniform',  # type_k1
            'uniform',  # type_k2
        ),
        (  # (13) stack of stars, (2,2,4) other, 3 coil, trajectory is different along second other dimension
            (2, 2, 4, 3, 8, 6, 6),  # im_shape
            (2, 2, 4, 3, 12, 6, 10),  # k_shape
            (2, 1, 1, 6, 10),  # nkx
            (2, 1, 1, 6, 10),  # nky
            (2, 1, 12, 1, 1),  # nkz
            'non-uniform',  # type_kx
            'non-uniform',  # type_ky
            'uniform',  # type_kz
            'non-uniform',  # type_k0
            'non-uniform',  # type_k1
            'uniform',  # type_k2
        ),
        (  # (14) 2d non-uniform not aligned in kzyx and k210, similar to 3d single shot spiral but with singleton dim
            (8, 5, 64, 1, 64),  # im_shape
            (8, 5, 1, 18, 128),  # k_shape
            (8, 1, 18, 128),  # nkx
            (8, 1, 1, 1),  # nky
            (8, 1, 18, 128),  # nkz
            'non-uniform',  # type_kx
            'zero',  # type_ky
            'non-uniform',  # type_kz
            'non-uniform',  # type_k0
            'non-uniform',  # type_k1
            'zero',  # type_k2
        ),
    ],
    ids=[
        '2d_cartesian_1_coil_no_oversampling',
        '2d_cartesian_1_coil_with_oversampling',
        '2d_non_cartesian_mri_2_coils',
        '2d_cartesian_irregular_sampling',
        '2d_single_shot_spiral',
        '3d_single_shot_spiral',
        '3d_nonuniform_4_coils_2_other',
        '2d_nonuniform_cine_mri_3_cardiac_phases_2_coils_same_traj',
        '2d_nonuniform_cine_mri_3_cardiac_phases_2_coils',
        '2d_cartesian_cine_2_cardiac_phases_3_coils',
        'radial_phase_encoding_3_coils_with_oversampling',
        'radial_phase_encoding_2_coils_non_cartesian_sampling',
        'stack_of_stars_3_other_4_coil_with_oversampling',
        'stack_of_stars_2_2_4_other',
        '2d_nonuniform_kzyx_k210_not_aligned',
    ],
)
