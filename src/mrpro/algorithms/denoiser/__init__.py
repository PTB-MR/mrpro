"""Pre-built denoising algorithms."""

from mrpro.algorithms.denoiser.total_variation_denoising import total_variation_denoising
from mrpro.algorithms.denoiser.patch_based_denoising import patch_based_denoising
__all__ = [
    'patch_based_denoising',
    'total_variation_denoising',
]
