"""Tests the IData class."""

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

import numpy as np
import pytest
import torch

from mrpro.data import IData
from tests.conftest import dcm_2d
from tests.conftest import dcm_multi_te
from tests.conftest import random_kheader
from tests.conftest import random_test_data


def test_IData_from_dcm_file(dcm_2d):
    """IData from single dicom file."""
    idat = IData.from_single_dicom(dcm_2d.filename)
    # IData uses complex values but dicom only supports real values
    im = torch.real(idat.data[0, 0, 0, ...])
    torch.testing.assert_close(im, dcm_2d.imref)


def test_IData_from_dcm_folder(dcm_multi_te):
    """IData from multiple dcm files in folder."""
    idat = IData.from_dicom_folder(dcm_multi_te[0].filename.parent)
    # Verify correct echo times
    te_orig = [ds.te for ds in dcm_multi_te]
    assert np.all(np.sort(te_orig) == np.sort(idat.header.te))
    # Verify all images were read in
    assert idat.data.shape[0] == len(te_orig)


def test_IData_from_kheader_and_tensor(random_kheader, random_test_data):
    """IData from KHeader and data tensor."""
    idat = IData.from_tensor_and_kheader(data=random_test_data, kheader=random_kheader)
    assert idat.header.te == random_kheader.te


def test_IData_to_complex128(random_kheader, random_test_data):
    """Change IData dtype complex128."""
    idat = IData.from_tensor_and_kheader(data=random_test_data, kheader=random_kheader)
    idat_complex128 = idat.to(dtype=torch.complex128)
    assert idat_complex128.data.dtype == torch.complex128


@pytest.mark.cuda
def test_IData_cuda(random_kheader, random_test_data):
    """Move IData object to CUDA memory."""
    idat = IData.from_tensor_and_kheader(data=random_test_data, kheader=random_kheader)
    idat_cuda = idat.cuda()
    assert idat_cuda.data.is_cuda


@pytest.mark.cuda
def test_IData_cpu(random_kheader, random_test_data):
    """Move IData object to CUDA memory and back to CPU memory."""
    idat = IData.from_tensor_and_kheader(data=random_test_data, kheader=random_kheader)
    idat_cpu = idat.cuda().cpu()
    assert idat_cpu.data.is_cpu
