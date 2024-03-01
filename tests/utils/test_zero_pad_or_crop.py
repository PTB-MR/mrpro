"""Tests for zero padding and cropping of data tensors."""

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

import torch
from mrpro.utils._zero_pad_or_crop import zero_pad_or_crop

from tests import RandomGenerator


def test_zero_pad_or_crop_content():
    """Test changing data by cropping and padding."""
    generator = RandomGenerator(seed=0)
    dshape_orig = (100, 200, 50)
    dshape_new = (80, 100, 240)
    dorig = generator.complex64_tensor(dshape_orig)
    dnew = zero_pad_or_crop(dorig, dshape_new, dim=(-3, -2, -1))

    # Compare overlapping region
    torch.testing.assert_close(dorig[10:90, 50:150, :], dnew[:, :, 95:145])
