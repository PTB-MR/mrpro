"""Tests for image space - k-space transformations."""

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

from mrpro.utils._change_data_shape import change_data_shape


def test_change_data_shape_content():
    """Test changing data size by cropping and padding."""
    dshape_orig = (100, 200, 50)
    dshape_new = (80, 100, 240)
    dorig = torch.randn(*dshape_orig, dtype=torch.complex64)
    dnew = change_data_shape(dorig, dshape_new)

    # Compare overlapping region
    torch.testing.assert_close(dorig[10:90, 50:150, :], dnew[:, :, 95:145])
