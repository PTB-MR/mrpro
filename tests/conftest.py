import tempfile

import ismrmrd
import pytest
from ismrmrd import xsd
from xsdata.models.datatype import XmlDate
from xsdata.models.datatype import XmlTime

from mrpro.data import AcqInfo
from mrpro.data import KHeader
from mrpro.data.enums import AcqFlags
from tests import RandomGenerator
from tests.data import Dicom2DTestImage
from tests.phantoms.test_ellipse_phantom import ph_ellipse


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


def generate_random_data(
    generator: RandomGenerator,
    shape=(32, 256),
):
    return generator.complex64_tensor(shape)


@pytest.fixture(params=({'seed': 0, 'Ncoils': 32, 'Nsamples': 256},))
def random_acquisition(request):
    seed, Ncoils, Nsamples = request.param['seed'], request.param['Ncoils'], request.param['Nsamples']
    generator = RandomGenerator(seed)
    kdata = generate_random_data(generator, (Ncoils, Nsamples))
    traj = generate_random_trajectory(generator, (Nsamples, 2))
    header = generate_random_acquisition_properties(generator)
    header['flags'] &= ~AcqFlags.ACQ_IS_NOISE_MEASUREMENT.value
    return ismrmrd.Acquisition.from_array(kdata, traj, **header)


@pytest.fixture(params=({'seed': 1, 'Ncoils': 32, 'Nsamples': 256},))
def random_noise_acquisition(request):
    seed, Ncoils, Nsamples = request.param['seed'], request.param['Ncoils'], request.param['Nsamples']
    generator = RandomGenerator(seed)
    kdata = generate_random_data(generator, (Ncoils, Nsamples))
    traj = generate_random_trajectory(generator, (Nsamples, 2))
    header = generate_random_acquisition_properties(generator)
    header['flags'] |= AcqFlags.ACQ_IS_NOISE_MEASUREMENT.value
    return ismrmrd.Acquisition.from_array(kdata, traj, **header)


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
    measurementInformation = xsd.measurementInformationType(
        measurementID=generator.ascii(10),
        seriesDate=XmlDate(generator.uint16(1970, 2030), generator.uint8(1, 12), generator.uint8(0, 30)),
        seriesTime=XmlTime(generator.uint8(0, 23), generator.uint8(0, 59), generator.uint8(0, 59)),
        sequenceName=generator.ascii(10),
    )

    acquisitionSystemInformation = xsd.acquisitionSystemInformationType(
        systemFieldStrength_T=generator.float32(0, 12),
        systemVendor=generator.ascii(10),
        systemModel=generator.ascii(10),
        receiverChannels=generator.uint16(1, 32),
    )

    sequenceParameters = xsd.sequenceParametersType(
        TR=[generator.float32()],
        TE=[generator.float32()],
        flipAngle_deg=[generator.float32(low=10, high=90)],
        echo_spacing=[generator.float32()],
        sequence_type=generator.ascii(10),
    )

    # TODO: add everything that to the header
    return xsd.ismrmrdschema.ismrmrdHeader(
        encoding=[encoding],
        sequenceParameters=sequenceParameters,
        version=generator.int16(),
        experimentalConditions=xsd.experimentalConditionsType(H1resonanceFrequency_Hz=generator.int32()),
        measurementInformation=measurementInformation,
        acquisitionSystemInformation=acquisitionSystemInformation,
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
    experimentalConditions = xsd.experimentalConditionsType(H1resonanceFrequency_Hz=generator.int32())
    return xsd.ismrmrdschema.ismrmrdHeader(
        encoding=[encoding],
        experimentalConditions=experimentalConditions,
    )


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
    ktraj = generate_random_trajectory(generator)
    kheader = KHeader.from_ismrmrd(random_full_ismrmrd_header, acq_info=random_acq_info, defaults={'trajectory': ktraj})
    return kheader


@pytest.fixture(params=({'seed': 0, 'Nother': 2, 'Ncoils': 16, 'Nz': 32, 'Ny': 128, 'Nx': 256},))
def random_test_data(request):
    seed, Nother, Ncoils, Nz, Ny, Nx = (
        request.param['seed'],
        request.param['Nother'],
        request.param['Ncoils'],
        request.param['Nz'],
        request.param['Ny'],
        request.param['Nx'],
    )
    generator = RandomGenerator(seed)
    test_data = generate_random_data(generator, (Nother, Ncoils, Nz, Ny, Nx))
    return test_data


@pytest.fixture(scope='session')
def dcm_2d(ph_ellipse, tmp_path_factory):
    """Single 2D dicom image."""
    dcm_filename = tmp_path_factory.mktemp('mrpro') / 'dicom_2d.dcm'
    dcm_idat = Dicom2DTestImage(filename=dcm_filename, phantom=ph_ellipse.phantom)
    return dcm_idat


@pytest.fixture(scope='session', params=({'num_images': 7},))
def dcm_multi_te(request, ph_ellipse, tmp_path_factory):
    """Multiple 2D dicom images with different echo times."""
    num_images = request.param['num_images']
    path = tmp_path_factory.mktemp('mrpro_multi_dcm')
    te = 2.0
    dcm_idat = []
    for _ in range(num_images):
        dcm_filename = path / f'dicom_te_{int(te)}.dcm'
        dcm_idat.append(Dicom2DTestImage(filename=dcm_filename, phantom=ph_ellipse.phantom, te=te))
        te += 1.0
    return dcm_idat
