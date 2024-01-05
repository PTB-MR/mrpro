"""Tests MotionOp class."""

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

from mrpro.data import DisplacementField
from mrpro.operators import MotionOp
from tests import RandomGenerator


@pytest.mark.parametrize(
    'im_shape, displacement_shape',
    [
        # One image, one motion state
        (
            (1, 1, 40, 50, 60),
            (1, 1, 40, 50, 60),
        ),
        # Several images each with the same motion state
        (
            (1, 5, 40, 50, 60),
            (1, 1, 40, 50, 60),
        ),
        # Several images each with a different motion state
        (
            (3, 1, 40, 50, 60),
            (3, 1, 40, 50, 60),
        ),
        # Several images sometimes with the same and sometimes with different motion states
        (
            (3, 5, 40, 50, 60),
            (3, 1, 40, 50, 60),
        ),
    ],
)
def test_different_shapes(im_shape, displacement_shape):
    """Test transformation with different shapes."""
    random_generator = RandomGenerator(seed=0)

    # Create image
    im = random_generator.complex64_tensor(size=im_shape)

    # Create displacement fields
    fz = random_generator.float32_tensor(size=displacement_shape)
    fy = random_generator.float32_tensor(size=displacement_shape)
    fx = random_generator.float32_tensor(size=displacement_shape)
    disp_field = DisplacementField(fz=fz, fy=fy, fx=fx)

    # Motion operator
    mop = MotionOp(disp_field)
    im_transformed = mop.forward(im)

    assert im.shape == im_transformed.shape


def test_translation():
    """Test translation motion transformation."""
    im_shape = [1, 1, 40, 60, 80]
    im_shift = [5, 10, 20]

    # Create images
    orig_im = torch.zeros(im_shape, dtype=torch.complex64)
    orig_im[:, :, 10:30, 15:45, 20:50] = 1 + 2j
    translated_im = torch.roll(orig_im, shifts=im_shift, dims=(-3, -2, -1))

    # Create transformation which shifts an object along z, y and x, respectively.
    fz = torch.ones(im_shape) * im_shift[0]
    fy = torch.ones(im_shape) * im_shift[1]
    fx = torch.ones(im_shape) * im_shift[2]
    disp_field = DisplacementField(fz=fz, fy=fy, fx=fx)

    # Create motion operator
    mop = MotionOp(disp_field)

    # Transform image
    im = mop.forward(orig_im)

    torch.testing.assert_close(translated_im, im)
