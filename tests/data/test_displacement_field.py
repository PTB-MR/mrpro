"""Tests for DisplacementField class."""

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
from einops import rearrange
from einops import repeat

from mrpro.data import DisplacementField
from tests import RandomGenerator


@pytest.mark.parametrize('nms,nother,nz,ny,nx', [(1, 1, 40, 16, 20), (6, 1, 40, 1, 20)])
def test_displacement_field_creation(nms, nother, nz, ny, nx):
    """Test the creation of displacement fields."""
    random_generator = RandomGenerator(seed=0)
    fz = random_generator.float32_tensor(size=(nms, nother, nz, ny, nx))
    fy = random_generator.float32_tensor(size=(nms, nother, nz, ny, nx))
    fx = random_generator.float32_tensor(size=(nms, nother, nz, ny, nx))

    disp_field = DisplacementField(fz=fz, fy=fy, fx=fx)

    torch.testing.assert_close(fz, disp_field.fz)
    torch.testing.assert_close(fy, disp_field.fy)
    torch.testing.assert_close(fx, disp_field.fx)


@pytest.mark.parametrize('nms,nother,nz,ny,nx', [(1, 1, 40, 16, 20), (6, 1, 40, 1, 20)])
def test_displacement_field_tensors(nms, nother, nz, ny, nx):
    """Test the creation of displacement fields tensors."""
    random_generator = RandomGenerator(seed=0)
    fz = random_generator.float32_tensor(size=(nms, nother, nz, ny, nx))
    fy = random_generator.float32_tensor(size=(nms, nother, nz, ny, nx))
    fx = random_generator.float32_tensor(size=(nms, nother, nz, ny, nx))
    f_tensor = torch.stack((fz, fy, fx), dim=0)

    disp_field = DisplacementField.from_tensor(f_tensor)

    torch.testing.assert_close(fz, disp_field.fz)
    torch.testing.assert_close(fy, disp_field.fy)
    torch.testing.assert_close(fx, disp_field.fx)
    torch.testing.assert_close(f_tensor, disp_field.as_tensor())
