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
from tests.helper import relative_image_difference
from tests.phantoms._EllipsePhantomTestData import EllipsePhantomTestData


def test_remove_readout_os(monkeypatch, random_kheader):
    # Dimensions
    n_coils = 4
    n_k0 = 240
    n_k1 = 240
    n_k0_oversampled = 320
    discard_pre = 10
    discard_post = 20

    random_generator = RandomGenerator(seed=0)

    # List of k1 indices in the shape
    idx_k1 = torch.arange(n_k1, dtype=torch.int32)[None, None, ...]

    # Set parameters need in remove_os
    monkeypatch.setattr(random_kheader.encoding_matrix, 'x', n_k0_oversampled)
    monkeypatch.setattr(random_kheader.recon_matrix, 'x', n_k0)
    monkeypatch.setattr(random_kheader.acq_info, 'center_sample', torch.zeros_like(idx_k1) + n_k0_oversampled // 2)
    monkeypatch.setattr(random_kheader.acq_info, 'number_of_samples', torch.zeros_like(idx_k1) + n_k0_oversampled)
    monkeypatch.setattr(random_kheader.acq_info, 'discard_pre', torch.tensor(discard_pre, dtype=torch.int32))
    monkeypatch.setattr(random_kheader.acq_info, 'discard_post', torch.tensor(discard_post, dtype=torch.int32))

    # Create kspace and image with oversampling
    phantom_os = EllipsePhantomTestData(n_y=n_k1, n_x=n_k0_oversampled)
    kdata_os = phantom_os.phantom.kspace(phantom_os.ky, phantom_os.kx)
    img_dim = SpatialDimension(z=1, y=n_k1, x=n_k0_oversampled)
    idata = phantom_os.phantom.image_space(img_dim)

    # Crop image data
    start = (n_k0_oversampled - n_k0) // 2
    idata = idata[..., start : start + n_k0]

    # Create k-space data with correct dimensions
    kdata = repeat(kdata_os, 'k1 k0 -> other coils k2 k1 k0', other=1, coils=n_coils, k2=1)

    # Create random 2D Cartesian trajectory
    kx = random_generator.float32_tensor(size=(1, 1, 1, n_k0_oversampled))
    ky = random_generator.float32_tensor(size=(1, 1, n_k1, 1))
    kz = random_generator.float32_tensor(size=(1, 1, 1, 1))
    trajectory = KTrajectory(kz, ky, kx)

    # Create KData
    kdata = KData(header=random_kheader, data=kdata, traj=trajectory)

    # Remove oversampling
    kdata = remove_readout_os(kdata)

    # Reconstruct image from k-space data of one coil and compare to phantom image
    fourier_op = FastFourierOp(dim=(-1, -2))
    (idata_recon,) = fourier_op.adjoint(kdata.data[:, 0, ...])

    # Due to discretisation artifacts the reconstructed image will be different to the reference image. Using standard
    # testing functions such as numpy.testing.assert_almost_equal fails because there are few voxels with high
    # differences along the edges of the elliptic objects.
    assert relative_image_difference(torch.abs(idata_recon), idata[:, 0, ...]) <= 0.05
