"""Tests for KTrajectory classes."""

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

from mrpro.data import KTrajectoryRpe
from mrpro.data import KTrajectorySunflowerGoldenRpe
from tests.data.conftest import random_kheader


def test_KTrajectoryRpe(kheader):
    """Calculate RPE trajectory."""
    ktrajectory = KTrajectoryRpe(angle=np.pi * 0.618034)
    ktraj = ktrajectory.calc_traj(kheader)
    assert ktraj is not None
