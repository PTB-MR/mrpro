"""Pre-built denoising algorithms."""

from mrpro.algorithms.denoiser.total_variation_denoising import total_variation_denoising
from mrpro.algorithms.denoiser.wavelet_denoising import wavelet_denoising
__all__ = [
    'total_variation_denoising',
    'wavelet_denoising'
]
