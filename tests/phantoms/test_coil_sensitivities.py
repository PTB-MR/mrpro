"""Tests for simulation of coil sensitivities."""

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
