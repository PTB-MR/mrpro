"""Remove oversampling along readout."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy

import torch

from mrpro.data import KData
from mrpro.data import KTrajectory
from mrpro.operators import FastFourierOp


def remove_readout_os(kdata: KData) -> KData:
    """Remove any oversampling along the readout (k0) direction.

    This function is inspired by https://github.com/gadgetron/gadgetron-python.
    Returns a copy of the data.

    Parameters
    ----------
    kdata
        K-space data

    Returns
    -------
        Copy of K-space data with oversampling removed.

    Raises
    ------
    ValueError
        If the recon matrix along x is larger than the encoding matrix along x.
    """
    # Ratio of k0/x between encoded and recon space
    x_ratio = kdata.header.recon_matrix.x / kdata.header.encoding_matrix.x
    if x_ratio == 1:
        # If the encoded and recon space is the same we don't have to do anything
        return deepcopy(kdata)
    elif x_ratio > 1:
        raise ValueError('Recon matrix along x should be equal or larger than encoding matrix along x.')

    # Starting and end point of image after removing oversampling
    start_cropped_readout = (kdata.header.encoding_matrix.x - kdata.header.recon_matrix.x) // 2
    end_cropped_readout = start_cropped_readout + kdata.header.recon_matrix.x

    def crop_readout(data_to_crop: torch.Tensor):
        # returns a cropped copy
        return data_to_crop[..., start_cropped_readout:end_cropped_readout].clone()

    # Transform to image space along readout, crop to reconstruction matrix size and transform back
    fourier_k0_op = FastFourierOp(dim=(-1,))
    (cropped_data,) = fourier_k0_op(crop_readout(*fourier_k0_op.H(kdata.data)))

    # Adapt trajectory
    ks = [kdata.traj.kz, kdata.traj.ky, kdata.traj.kx]
    # only cropped ks that are not broadcasted/singleton along k0
    cropped_ks = [crop_readout(k) if k.shape[-1] > 1 else k.clone() for k in ks]
    cropped_traj = KTrajectory(*cropped_ks)

    # Adapt header parameters
    header = deepcopy(kdata.header)
    header.acq_info.center_sample -= start_cropped_readout
    header.acq_info.number_of_samples[:] = cropped_data.shape[-1]
    header.encoding_matrix.x = cropped_data.shape[-1]

    header.acq_info.discard_post = (header.acq_info.discard_post * x_ratio).to(torch.int32)
    header.acq_info.discard_pre = (header.acq_info.discard_pre * x_ratio).to(torch.int32)

    return KData(header, cropped_data, cropped_traj)
