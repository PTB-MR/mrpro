"""Prewhiten k-space data."""

from copy import deepcopy

import torch
from einops import einsum, parse_shape, rearrange

from mrpro.data._kdata.KData import KData
from mrpro.data.KNoise import KNoise


def prewhiten_kspace(kdata: KData, knoise: KNoise, scale_factor: float | torch.Tensor = 1.0) -> KData:
    """Calculate noise prewhitening matrix and decorrelate coils.

    Steps:

    - Calculate noise correlation matrix N
    - Carry out Cholesky decomposition L L^H = N
    - Estimate noise decorrelation matrix D = inv(L)
    - Apply D to k-space data

    More information can be found in [ISMa]_ [HAN2014]_ [ROE1990]_.

    If the data has more samples in the 'other'-dimensions (batch/slice/...),
    the noise covariance matrix is calculated jointly over all samples.
    Thus, if the noise is not stationary, the noise covariance matrix is not accurate.
    In this case, the function should be called for each sample separately.

    Parameters
    ----------
    kdata
        K-space data.
    knoise
        Noise measurements.
    scale_factor
        Square root is applied on the noise covariance matrix. Used to adjust for effective noise bandwidth
        and difference in sampling rate between noise calibration and actual measurement:
        scale_factor = (T_acq_dwell/T_noise_dwell)*NoiseReceiverBandwidthRatio

    Returns
    -------
        Prewhitened copy of k-space data

    References
    ----------
    .. [ISMa] ISMRMRD Python tools https://github.com/ismrmrd/ismrmrd-python-tools
    .. [HAN2014] Hansen M, Kellman P (2014) Image reconstruction: An overview for clinicians. JMRI 41(3)
            https://doi.org/10.1002/jmri.24687
    .. [ROE1990] Roemer P, Mueller O (1990) The NMR phased array. MRM 16(2)
            https://doi.org/10.1002/mrm.1910160203
    """
    # Reshape noise to (coil, everything else)
    noise = rearrange(knoise.data, '... coils k2 k1 k0->coils (... k2 k1 k0)')

    # Calculate noise covariance matrix and Cholesky decomposition
    noise_cov = (1.0 / (noise.shape[-1])) * einsum(
        noise, noise.conj(), 'coil1 everythingelse, coil2 everythingelse -> coil1 coil2'
    )

    cholsky = torch.linalg.cholesky(noise_cov)

    # solve_triangular is numerically more stable than inverting the matrix
    # but requires a single batch dimension
    kdata_flat = rearrange(kdata.data, '... coil k2 k1 k0 -> (... k2 k1 k0) coil 1')
    whitened_flat = torch.linalg.solve_triangular(cholsky, kdata_flat, upper=False)
    whitened_flatother = rearrange(
        whitened_flat, '(other k2 k1 k0) coil 1-> other coil k2 k1 k0', **parse_shape(kdata.data, '... k2 k1 k0')
    )
    whitened_data = whitened_flatother.reshape(kdata.data.shape) * torch.as_tensor(scale_factor).sqrt()
    header = deepcopy(kdata.header)
    traj = deepcopy(kdata.traj)

    return KData(header, whitened_data, traj)
