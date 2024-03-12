"""Tests for simulation of coil sensitivities."""

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

from mrpro.data import SpatialDimension
from mrpro.phantoms.coils import birdcage_2d


def test_birdcage_sensitivities_shape():
    n_z = 1
    n_y = 200
    n_x = 150
    n_coils = 4
    image_dimension = SpatialDimension(z=n_z, y=n_y, x=n_x)
    simulated_coil_sensitivities = birdcage_2d(n_coils, image_dimension)
    assert simulated_coil_sensitivities.shape == (1, n_coils, n_z, n_y, n_x)
