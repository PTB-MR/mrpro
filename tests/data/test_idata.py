"""Tests the IData class."""

from pathlib import Path

import pytest
import torch
from mr2.data import IData


@pytest.mark.parametrize(
    ('dcm_data_fixture'),
    [
        'dcm_2d',
        'dcm_2d_with_empty_field',
        'dcm_3d',
        'dcm_2d_rescale',
        'dcm_3d_rescale',
        'dcm_2d_multi_echo_times',
        'dcm_2d_multi_echo_times_multi_folders',
        'dcm_cardiac_2d',
        'dcm_m2d_multi_orientation',
        'dcm_3d_multi_echo',
        'dcm_3d_multi_echo_multi_cardiac_phases',
        'dcm_3d_multi_orientation',
    ],
)
def test_IData_content_from_dcm(dcm_data_fixture, request):
    """Verify image content from different dicom types."""
    dcm_data = request.getfixturevalue(dcm_data_fixture)
    idata = IData.from_dicom_folder(dcm_data[0].filename.parent)
    # IData uses complex values but dicom only supports real values
    first_img = torch.real(idata.data.flatten(end_dim=-3)[0])
    # Rescaling can lead to loss of precision due to uint16 storage
    torch.testing.assert_close(
        first_img, dcm_data[0].img_ref, atol=dcm_data[0].rescale_slope, rtol=dcm_data[0].rescale_slope / (2**16 - 1)
    )


def test_IData_from_dcm_file(dcm_2d):
    """IData from dicom file."""
    idata = IData.from_dicom_files(dcm_2d[0].filename)
    # IData uses complex values but dicom only supports real values
    img = torch.real(idata.data[0, 0, 0, ...])
    torch.testing.assert_close(img, dcm_2d[0].img_ref)


@pytest.mark.parametrize('magnitude_only', [True, False])
def test_IData_to_nifti(dcm_2d, tmp_path, magnitude_only: bool) -> None:
    """Save image data as NIFTI 1/2 file."""
    idata = IData.from_dicom_files(dcm_2d[0].filename)
    idata.to_nifti(tmp_path / 'test.nii', magnitude_only=magnitude_only)
    assert (tmp_path / 'test.nii').exists()


@pytest.mark.parametrize(
    ('dcm_data_fixture'),
    [
        'dcm_2d_multi_echo_times',
        'dcm_3d_multi_echo',
        'dcm_3d_multi_echo_multi_cardiac_phases',
    ],
)
def test_IData_from_multi_echo_dicom(dcm_data_fixture, request):
    """IData from multiple dcm files in folder."""
    dcm_data = request.getfixturevalue(dcm_data_fixture)
    idata = IData.from_dicom_folder(dcm_data[0].filename.parent)
    # Verify correct echo times
    original_echo_times = torch.as_tensor([ds.te for ds in dcm_data])
    assert idata.header.te is not None
    assert torch.allclose(torch.sort(original_echo_times)[0], torch.sort(torch.as_tensor(idata.header.te))[0])
    # Verify all images were read in
    assert idata.data.shape[0] == original_echo_times.shape[0]


def test_IData_from_dcm_folder_via_path(dcm_2d_multi_echo_times):
    """IData from multiple dcm files in folder."""
    idata = IData.from_dicom_files(Path(dcm_2d_multi_echo_times[0].filename.parent).glob('*.dcm'))
    # Verify correct echo times
    original_echo_times = torch.as_tensor([ds.te for ds in dcm_2d_multi_echo_times])
    assert idata.header.te is not None
    assert torch.allclose(torch.sort(original_echo_times)[0], torch.sort(torch.as_tensor(idata.header.te))[0])
    # Verify all images were read in
    assert idata.data.shape[0] == len(original_echo_times)


def test_IData_from_wrong_path():
    """Error for non-existing/empty folder/wrong suffix."""
    with pytest.raises(ValueError, match='No dicom files with suffix'):
        _ = IData.from_dicom_folder('non/existing/path')


def test_IData_from_empty_dcm_file_list():
    """Error for empty file list."""
    with pytest.raises(ValueError, match='No dicom files specified'):
        _ = IData.from_dicom_files([])


def test_IData_from_dcm_files(dcm_2d_multi_echo_times_multi_folders):
    """IData from multiple dcm files in different folders."""
    idata = IData.from_dicom_files([dcm_file.filename for dcm_file in dcm_2d_multi_echo_times_multi_folders])
    # Verify correct echo times
    original_echo_times = torch.as_tensor([ds.te for ds in dcm_2d_multi_echo_times_multi_folders])
    assert idata.header.te is not None
    assert torch.allclose(torch.sort(original_echo_times)[0], torch.sort(torch.as_tensor(idata.header.te))[0])
    # Verify all images were read in
    assert idata.data.shape[0] == len(original_echo_times)


def test_IData_from_kheader_and_tensor(random_kheader, random_test_data):
    """IData from KHeader and data tensor."""
    random_kheader.ti = []
    idata = IData.from_tensor_and_kheader(data=random_test_data, header=random_kheader)
    assert idata.header.te == random_kheader.te
    assert idata.header.ti == random_kheader.ti
    assert idata.header.te is not random_kheader.te
    torch.testing.assert_close(idata.data, random_test_data)


def test_IData_to_complex128(random_kheader, random_test_data):
    """Change IData dtype complex128."""
    idata = IData.from_tensor_and_kheader(data=random_test_data, header=random_kheader)
    idata_complex128 = idata.to(dtype=torch.complex128)
    assert idata_complex128.data.dtype == torch.complex128


@pytest.mark.cuda
def test_IData_cuda(random_kheader, random_test_data):
    """Move IData object to CUDA memory."""
    idata = IData.from_tensor_and_kheader(data=random_test_data, header=random_kheader)
    idata_cuda = idata.cuda()
    assert idata_cuda.data.is_cuda


@pytest.mark.cuda
def test_IData_cpu(random_kheader, random_test_data):
    """Move IData object to CUDA memory and back to CPU memory."""
    idata = IData.from_tensor_and_kheader(data=random_test_data, header=random_kheader)
    idata_cpu = idata.cuda().cpu()
    assert idata_cpu.data.is_cpu


def test_IData_rss(random_kheader, random_test_data):
    """Test RSS coil combination."""
    expected = random_test_data.abs().square().sum(dim=-4, keepdim=True).sqrt()
    idata = IData.from_tensor_and_kheader(data=random_test_data, header=random_kheader)
    torch.testing.assert_close(idata.rss(keepdim=True), expected)
    torch.testing.assert_close(idata.rss(keepdim=False), expected.squeeze(-4))


@pytest.mark.parametrize(('dcm_data_fixture'), ['dcm_2d', 'dcm_3d'])
def test_IData_to_dicom_folder_identical(dcm_data_fixture, request):
    """Verify saving of different dicom types."""
    dcm_data = request.getfixturevalue(dcm_data_fixture)
    idata = IData.from_dicom_folder(dcm_data[0].filename.parent)
    idata.to_dicom_folder(dcm_data[0].filename.parent / 'test_output', series_description='test_series')
    idata_reloaded = IData.from_dicom_folder(dcm_data[0].filename.parent / 'test_output')

    torch.testing.assert_close(idata_reloaded.header.te, idata.header.te)
    torch.testing.assert_close(idata_reloaded.header.tr, idata.header.tr)
    torch.testing.assert_close(idata_reloaded.header.fa, idata.header.fa)
    torch.testing.assert_close(idata_reloaded.header.ti, idata.header.ti)

    torch.testing.assert_close(idata_reloaded.header.position.x, idata.header.position.x)
    torch.testing.assert_close(idata_reloaded.header.position.y, idata.header.position.y)
    torch.testing.assert_close(idata_reloaded.header.position.z, idata.header.position.z)

    assert idata_reloaded.header.orientation == idata.header.orientation
    torch.testing.assert_close(idata_reloaded.data, idata.data)


@pytest.mark.parametrize(
    ('dcm_data_fixture'),
    [
        'dcm_2d_multi_echo_times',
        'dcm_3d_multi_echo',
        'dcm_3d_multi_echo_multi_cardiac_phases',
    ],
)
def test_IData_to_dicom_folder(dcm_data_fixture, request):
    """Verify saving of different dicom types."""
    dcm_data = request.getfixturevalue(dcm_data_fixture)
    idata = IData.from_dicom_folder(dcm_data[0].filename.parent)
    with pytest.warns(UserWarning, match='is not singleton. Using first value.'):
        idata.to_dicom_folder(dcm_data[0].filename.parent / 'test_output', series_description='test_series')
    idata_reloaded = IData.from_dicom_folder(dcm_data[0].filename.parent / 'test_output')

    torch.testing.assert_close(idata_reloaded.header.te[0], idata.header.te[0])
    torch.testing.assert_close(idata_reloaded.header.position.x, idata.header.position.x)
    torch.testing.assert_close(idata_reloaded.header.position.y, idata.header.position.y)
    torch.testing.assert_close(idata_reloaded.header.position.z, idata.header.position.z)
    assert idata_reloaded.header.orientation == idata.header.orientation
    torch.testing.assert_close(idata_reloaded.data, idata.data)


@pytest.mark.parametrize(('rescale_slope'), [10, 20])
@pytest.mark.parametrize(('rescale_intercept'), [-100, -200])
def test_IData_to_dicom_folder_with_scaling(tmp_path_factory, dcm_2d, rescale_slope, rescale_intercept):
    """Verify data is correctly scaled during saving."""
    idata = IData.from_dicom_folder(dcm_2d[0].filename.parent)
    dicom_folder = tmp_path_factory.mktemp('dicom_scaling') / 'test_output'
    idata.to_dicom_folder(
        dicom_folder,
        series_description='test_series',
        rescale_slope=rescale_slope,
        rescale_intercept=rescale_intercept,
        normalize_data=True,
    )
    idata_reloaded = IData.from_dicom_folder(dicom_folder)
    torch.testing.assert_close(idata_reloaded.data, idata.data, atol=rescale_slope, rtol=rescale_slope / (2**16 - 1))


def test_IData_to_dicom_folder_warning_cropped(tmp_path_factory, dcm_2d):
    """Warning if data is cropped"""
    idata = IData.from_dicom_folder(dcm_2d[0].filename.parent)
    dicom_folder = tmp_path_factory.mktemp('dicom_warning') / 'test_output'
    with pytest.warns(UserWarning, match='Values outside of the uint16 range will be clipped.'):
        idata.to_dicom_folder(
            dicom_folder,
            series_description='test_series',
            rescale_slope=0.1,
            rescale_intercept=100,
            normalize_data=False,
        )
    idata_reloaded = IData.from_dicom_folder(dicom_folder)
    assert idata_reloaded.header.orientation == idata.header.orientation


def test_IData_from_kheader_and_tensor_to_dicom_folder(tmp_path_factory, random_kheader, random_test_data):
    """IData from KHeader and data tensor."""
    dicom_folder = tmp_path_factory.mktemp('dicom_from_kheader_and_tensor2') / 'test_output'
    idata = IData.from_tensor_and_kheader(data=random_test_data, header=random_kheader)  # shape: [2, 8, 16, 32, 64]
    idata.to_dicom_folder(dicom_folder, series_description='test_series')
    idata_reloaded = IData.from_dicom_folder(dicom_folder)  # shape: [16, 1, 16, 32, 64]

    torch.testing.assert_close(idata_reloaded.header.te[0], idata.header.te[0])
    torch.testing.assert_close(idata_reloaded.header.position.x, idata.header.position.x)
    torch.testing.assert_close(idata_reloaded.header.position.y, idata.header.position.y)
    torch.testing.assert_close(idata_reloaded.header.position.z, idata.header.position.z)
    assert idata_reloaded.header.orientation == idata.header.orientation

    # Compare image pixel values
    other, coil = idata.shape[:2]
    idata_reloaded_ra = idata_reloaded.rearrange('(other coil) 1 z y x -> other coil z y x', other=other, coil=coil)
    idata_reloaded_px = idata_reloaded_ra.data.abs() / (2**16 - 1)
    idata_px = idata.data.abs()

    torch.testing.assert_close(idata_px, idata_reloaded_px, rtol=1e-4, atol=1e-4)
