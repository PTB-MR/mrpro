"""Tests the QData class."""

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
from mrpro.data import IHeader, QData


def test_QData_from_kheader_and_tensor(random_kheader, random_test_data):
    """QData from KHeader and data tensor."""
    qdata = QData(data=random_test_data, header=random_kheader)
    assert qdata.header.fov == random_kheader.recon_fov


def test_QData_from_iheader_and_tensor(random_kheader, random_test_data):
    """QData from IHeader (created from KHeader) and data tensor."""
    iheader_from_kheader = IHeader.from_kheader(random_kheader)
    qdata = QData(data=random_test_data, header=iheader_from_kheader)
    assert qdata.header.fov == iheader_from_kheader.fov


def test_QData_from_dcm_file(dcm_2d):
    """QData from single dicom file."""
    qdata = QData.from_single_dicom(dcm_2d.filename)
    # QData uses complex values but dicom only supports real values
    img = torch.real(qdata.data[0, 0, 0, ...])
    torch.testing.assert_close(img, dcm_2d.img_ref)


def test_QData_to_complex128(random_kheader, random_test_data):
    """Change IData dtype complex128."""
    qdata = QData(data=random_test_data, header=random_kheader)
    qdata_complex128 = qdata.to(dtype=torch.complex128)
    assert qdata_complex128.data.dtype == torch.complex128


@pytest.mark.cuda()
def test_QData_cuda(random_kheader, random_test_data):
    """Move IData object to CUDA memory."""
    qdata = QData(data=random_test_data, header=random_kheader)
    qdata_cuda = qdata.cuda()
    assert qdata_cuda.data.is_cuda


@pytest.mark.cuda()
def test_QData_cpu(random_kheader, random_test_data):
    """Move IData object to CUDA memory and back to CPU memory."""
    qdata = QData(data=random_test_data, header=random_kheader)
    qdata_cpu = qdata.cuda().cpu()
    assert qdata_cpu.data.is_cpu
