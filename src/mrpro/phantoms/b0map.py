"""Random B0 map generation."""

import torch
from scipy.special import sph_harm_y

from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.RandomGenerator import RandomGenerator


def random_b0map(
    shape: SpatialDimension[int],
    fov: SpatialDimension[float],
    l_max: int = 3,
    sigma_ppm: float = 1000.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Simulate B0 inhomogeneity map via randomized spherical harmonics.

    Parameters
    ----------
    shape
        Grid dimensions
    fov
        Field of view in meters (fov_z, fov_y, fov_x).
    l_max
        Maximum spherical harmonic degree.
    sigma_ppm
        Std of inhomogeneity in ppm.
    seed
        Random seed.

    Returns
    -------
    b0_map
        (z, y, x) B0 field map in ppm.
    """
    rng = RandomGenerator(seed)

    r_ref = max(fov.zyx) / 2
    z = torch.linspace(-fov.z / 2, fov.z / 2, shape.z)
    y = torch.linspace(-fov.y / 2, fov.y / 2, shape.y)
    x = torch.linspace(-fov.x / 2, fov.x / 2, shape.x)
    z, y, x = torch.meshgrid(z, y, x, indexing='ij')
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.arccos(torch.clamp(z / (r + 1e-12), -1, 1))
    phi = torch.atan2(y, x)

    b0_map = torch.zeros(shape.zyx, dtype=torch.float64)

    for n in range(1, l_max + 1):
        for m in range(-n, n + 1):
            if m > 0:
                y_real = (-1) ** m * (2**0.5) * sph_harm_y(n, m, phi.numpy(), theta.numpy()).real
            elif m == 0:
                y_real = sph_harm_y(n, 0, phi.numpy(), theta.numpy()).real
            else:
                y_real = (-1) ** m * (2**0.5) * sph_harm_y(n, -m, phi.numpy(), theta.numpy()).imag

            solid = (r / r_ref) ** n * y_real
            coeff = rng.randn_tensor((1,), torch.float64) * sigma_ppm / n
            b0_map += coeff * solid

    return b0_map
