"""Pre-built denoising algorithms."""

from mrpro.algorithms.denoiser.total_variation_denoising import total_variation_denoising
from mrpro.algorithms.denoiser.conv_analysis_dictionary_denoising import (
    conv_analysis_dictionary_denoising,
)
from mrpro.algorithms.denoiser.conv_synthesis_dictionary_denoising import (
    conv_synthesis_dictionary_denoising,
)
__all__ = [
    'conv_analysis_dictionary_denoising',
    'conv_synthesis_dictionary_denoising',
    'total_variation_denoising',
]
