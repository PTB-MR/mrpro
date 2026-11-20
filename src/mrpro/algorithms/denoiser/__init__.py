"""Pre-built denoising algorithms."""

from mrpro.algorithms.denoiser.total_variation_denoising import total_variation_denoising
from mrpro.algorithms.denoiser.wavelet_soft_thresholding import wavelet_soft_thresholding
from mrpro.algorithms.denoiser.transform_domain_soft_thresholding import transform_domain_soft_thresholding

__all__ = ['total_variation_denoising', 'transform_domain_soft_thresholding', 'wavelet_soft_thresholding']
