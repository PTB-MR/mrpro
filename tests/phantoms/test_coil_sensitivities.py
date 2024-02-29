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
    nz = 1
    ny = 200
    nx = 150
    num_coils = 4
    im_dim = SpatialDimension(z=nz, y=ny, x=nx)
    sim_coil = birdcage_2d(num_coils, im_dim)
    assert sim_coil.shape == (1, num_coils, nz, ny, nx)
