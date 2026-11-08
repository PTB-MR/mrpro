"""Pre-built denoising algorithms."""

from mrpro.algorithms.denoiser.total_variation_denoising import total_variation_denoising
from mrpro.algorithms.denoiser.domain_thresholding import domain_thresholding

__all__ = ['domain_thresholding', 'total_variation_denoising']
