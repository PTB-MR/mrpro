"""Tests the IData class."""

from pathlib import Path

import pytest
import torch
from mrpro.data import IData


@pytest.mark.parametrize(
    ('dcm_data_fixture'),
    [
        'dcm_2d',
        'dcm_3d',
        'dcm_2d_multi_echo_times',
        'dcm_2d_multi_echo_times_multi_folders',
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
    img = torch.real(idata.data[0, 0, 0, ...])
    torch.testing.assert_close(img, dcm_data[0].img_ref)


def test_IData_from_dcm_file(dcm_2d):
    """IData from dicom file."""
    idata = IData.from_dicom_files(dcm_2d[0].filename)
    # IData uses complex values but dicom only supports real values
    img = torch.real(idata.data[0, 0, 0, ...])
    torch.testing.assert_close(img, dcm_2d[0].img_ref)


def test_IData_save_as_nifti(dcm_2d, tmp_path):
    """Save image data as NIFTI2 file."""
    idata = IData.from_dicom_files(dcm_2d[0].filename)
    idata.save_as_nifti(tmp_path / 'test')
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
