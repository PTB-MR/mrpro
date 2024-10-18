"""Tests the IData class."""

from pathlib import Path

import pytest
import torch
from mrpro.data import IData


def test_IData_from_dcm_file(dcm_2d):
    """IData from single dicom file."""
    idata = IData.from_single_dicom(dcm_2d.filename)
    # IData uses complex values but dicom only supports real values
    img = torch.real(idata.data[0, 0, 0, ...])
    torch.testing.assert_close(img, dcm_2d.img_ref)


def test_IData_from_dcm_folder(dcm_multi_echo_times):
    """IData from multiple dcm files in folder."""
    idata = IData.from_dicom_folder(dcm_multi_echo_times[0].filename.parent)
    # Verify correct echo times
    original_echo_times = torch.as_tensor([ds.te for ds in dcm_multi_echo_times])
    assert idata.header.te is not None
    # dicom expects echo times in ms, mrpro in s
    assert torch.allclose(torch.sort(original_echo_times)[0] / 1000, torch.sort(idata.header.te)[0])
    # Verify all images were read in
    assert idata.data.shape[0] == original_echo_times.shape[0]


def test_IData_from_dcm_folder_via_path(dcm_multi_echo_times):
    """IData from multiple dcm files in folder."""
    idata = IData.from_dicom_files(Path(dcm_multi_echo_times[0].filename.parent).glob('*.dcm'))
    # Verify correct echo times
    original_echo_times = torch.as_tensor([ds.te for ds in dcm_multi_echo_times])
    assert idata.header.te is not None
    # dicom expects echo times in ms, mrpro in s
    assert torch.allclose(torch.sort(original_echo_times)[0] / 1000, torch.sort(idata.header.te)[0])
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


def test_IData_from_dcm_files(dcm_multi_echo_times_multi_folders):
    """IData from multiple dcm files in different folders."""
    idata = IData.from_dicom_files([dcm_file.filename for dcm_file in dcm_multi_echo_times_multi_folders])
    # Verify correct echo times
    original_echo_times = torch.as_tensor([ds.te for ds in dcm_multi_echo_times_multi_folders])
    assert idata.header.te is not None
    # dicom expects echo times in ms, mrpro in s
    assert torch.allclose(torch.sort(original_echo_times)[0] / 1000, torch.sort(idata.header.te)[0])
    # Verify all images were read in
    assert idata.data.shape[0] == len(original_echo_times)


def test_IData_from_kheader_and_tensor(random_kheader, random_test_data):
    """IData from KHeader and data tensor."""
    idata = IData.from_tensor_and_kheader(data=random_test_data, kheader=random_kheader)
    assert idata.header.te == random_kheader.te


def test_IData_to_complex128(random_kheader, random_test_data):
    """Change IData dtype complex128."""
    idata = IData.from_tensor_and_kheader(data=random_test_data, kheader=random_kheader)
    idata_complex128 = idata.to(dtype=torch.complex128)
    assert idata_complex128.data.dtype == torch.complex128


@pytest.mark.cuda
def test_IData_cuda(random_kheader, random_test_data):
    """Move IData object to CUDA memory."""
    idata = IData.from_tensor_and_kheader(data=random_test_data, kheader=random_kheader)
    idata_cuda = idata.cuda()
    assert idata_cuda.data.is_cuda


@pytest.mark.cuda
def test_IData_cpu(random_kheader, random_test_data):
    """Move IData object to CUDA memory and back to CPU memory."""
    idata = IData.from_tensor_and_kheader(data=random_test_data, kheader=random_kheader)
    idata_cpu = idata.cuda().cpu()
    assert idata_cpu.data.is_cpu


def test_IData_rss(random_kheader, random_test_data):
    """Test RSS coil combination."""
    expected = random_test_data.abs().square().sum(dim=-4, keepdim=True).sqrt()
    idata = IData.from_tensor_and_kheader(data=random_test_data, kheader=random_kheader)
    torch.testing.assert_close(idata.rss(keepdim=True), expected)
    torch.testing.assert_close(idata.rss(keepdim=False), expected.squeeze(-4))
