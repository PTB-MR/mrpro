"""Tests the Knoise class."""

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
from mrpro.data import KNoise


def test_knoise_to_complex128(random_test_data):
    """Change dtype to complex128."""
    noise = KNoise(data=random_test_data).to(dtype=torch.complex128)
    assert noise.data.dtype == torch.complex128


@pytest.mark.cuda()
def test_knoise_cuda(random_test_data):
    """Move KNois object to CUDA memory."""
    noise = KNoise(data=random_test_data).cuda()
    assert noise.data.is_cuda


@pytest.mark.cuda()
def test_knoise_cpu(random_test_data):
    """Move KNoise object to CUDA memory and back to CPU memory."""
    noise_cuda = KNoise(data=random_test_data).cuda()
    noise_cpu = noise_cuda.cpu()
    assert noise_cpu.data.is_cpu
    assert noise_cuda.data.is_cuda
    torch.testing.assert_close(noise_cpu.data, noise_cuda.data.cpu())
