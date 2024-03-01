"""Remove oversampling along readout."""

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

import torch

from mrpro.data import KData
from mrpro.data import KTrajectory
from mrpro.operators import FastFourierOp


def remove_readout_os(kdata: KData) -> KData:
    """Remove any oversampling along the readout (k0) direction.

    This function is inspired by https://github.com/gadgetron/gadgetron-python.

    Parameters
    ----------
    kdata
        K-space data

    Returns
    -------
        K-space data with oversampling removed.

    Raises
    ------
    ValueError
        If the recon matrix along x is larger than the encoding matrix along x.
    """
    # Ratio between encoded and recon space
    dim_ratio = kdata.header.recon_matrix.x / kdata.header.encoding_matrix.x

    # If the encoded and recon space is the same we don't have to do anything
    if dim_ratio == 1:
        return kdata
    elif dim_ratio > 1:
        raise ValueError('Recon matrix along x should be equal or larger than encoding matrix along x.')
    else:
        # Starting and end point of image after removing oversampling
        start_cropped_readout = (kdata.header.encoding_matrix.x - kdata.header.recon_matrix.x) // 2
        end_cropped_readout = start_cropped_readout + kdata.header.recon_matrix.x

        def crop_readout(input_):
            return input_[..., start_cropped_readout:end_cropped_readout]

        # Transform to image space, crop to reconstruction matrix size and transform back
        FFOp = FastFourierOp(dim=(-1,))
        (dat,) = FFOp.adjoint(kdata.data)
        dat = crop_readout(dat)
        (dat,) = FFOp.forward(dat)

        # Adapt trajectory
        ks = [kdata.traj.kz, kdata.traj.ky, kdata.traj.kx]
        for ax in range(3):
            if ks[ax].shape[-1] > 1:
                ks[ax] = crop_readout(ks[ax])
        traj = KTrajectory(kz=ks[0], ky=ks[1], kx=ks[2])

        # Adapt header parameters
        hdr = kdata.header
        hdr.acq_info.center_sample -= start_cropped_readout
        hdr.acq_info.number_of_samples[:] = dat.shape[-1]
        hdr.encoding_matrix.x = dat.shape[-1]

        hdr.acq_info.discard_post = (hdr.acq_info.discard_post * dim_ratio).to(torch.int32)
        hdr.acq_info.discard_pre = (hdr.acq_info.discard_pre * dim_ratio).to(torch.int32)

    return KData(hdr, dat, traj)
