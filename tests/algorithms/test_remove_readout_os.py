"""Test remove oversampling along readout."""

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
from einops import repeat
from mrpro.algorithms import remove_readout_os
from mrpro.data import KData
from mrpro.data import KTrajectory
from mrpro.data import SpatialDimension
from mrpro.operators import FastFourierOp
from tests import RandomGenerator
from tests.helper import rel_image_diff
from tests.phantoms._EllipsePhantomTestData import EllipsePhantomTestData


def test_remove_readout_os(monkeypatch, random_kheader):
    # Dimensions
    ncoils = 4
    nk0 = 240
    nk1 = 240
    nk0_os = 320
    discard_pre = 10
    discard_post = 20

    random_generator = RandomGenerator(seed=0)

    # List of k1 indices in the shape
    idx_k1 = torch.arange(nk1, dtype=torch.int32)[None, None, ...]

    # Set parameters need in remove_os
    monkeypatch.setattr(random_kheader.encoding_matrix, 'x', nk0_os)
    monkeypatch.setattr(random_kheader.recon_matrix, 'x', nk0)
    monkeypatch.setattr(random_kheader.acq_info, 'center_sample', torch.zeros_like(idx_k1) + nk0_os // 2)
    monkeypatch.setattr(random_kheader.acq_info, 'number_of_samples', torch.zeros_like(idx_k1) + nk0_os)
    monkeypatch.setattr(random_kheader.acq_info, 'discard_pre', torch.tensor(discard_pre, dtype=torch.int32))
    monkeypatch.setattr(random_kheader.acq_info, 'discard_post', torch.tensor(discard_post, dtype=torch.int32))

    # Create kspace and image with oversampling
    ph_os = EllipsePhantomTestData(ny=nk1, nx=nk0_os)
    kdat_os = ph_os.phantom.kspace(ph_os.ky, ph_os.kx)
    im_dim = SpatialDimension(z=1, y=nk1, x=nk0_os)
    idat = ph_os.phantom.image_space(im_dim)

    # Crop image data
    idat_start = (nk0_os - nk0) // 2
    idat = idat[..., idat_start : idat_start + nk0]

    # Create k-space data with correct dimensions
    kdat = repeat(kdat_os, 'k1 k0 -> other coils k2 k1 k0', other=1, coils=ncoils, k2=1)

    # Create random 2D Cartesian trajectory
    kx = random_generator.float32_tensor(size=(1, 1, 1, nk0_os))
    ky = random_generator.float32_tensor(size=(1, 1, nk1, 1))
    kz = random_generator.float32_tensor(size=(1, 1, 1, 1))
    ktraj = KTrajectory(kz, ky, kx)

    # Create KData
    kdata = KData(header=random_kheader, data=kdat, traj=ktraj)

    # Remove oversampling
    kdata = remove_readout_os(kdata)

    # Reconstruct image from k-space data of one coil and compare to phantom image
    FFOp = FastFourierOp(dim=(-1, -2))
    (idat_rec,) = FFOp.adjoint(kdata.data[:, 0, ...])

    # Due to discretisation artifacts the reconstructed image will be different to the reference image. Using standard
    # testing functions such as numpy.testing.assert_almost_equal fails because there are few voxels with high
    # differences along the edges of the elliptic objects.
    assert rel_image_diff(torch.abs(idat_rec), idat[:, 0, ...]) <= 0.05
