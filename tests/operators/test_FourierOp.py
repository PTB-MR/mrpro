import pytest
import torch

from mrpro.data import KTrajectory
from mrpro.data import SpatialDimension
from mrpro.operators import FourierOp
from tests import RandomGenerator


def create_uniform_traj(nk):
    """Create a tensor of uniform points with predefined shape nk."""
    if nk[1:].count(1) <= 1:
        raise ValueError('nk is allowed to have at most one non-singleton dimension')
    n_kpoints = torch.tensor(nk[1:]).max()
    if n_kpoints > 1:
        k = torch.linspace(-1, 1, n_kpoints)
        views = [1 if i != n_kpoints else -1 for i in nk]
        k = k.view(*views).expand(list(nk))
    else:
        k = torch.zeros(nk)
    return k


def create_data(im_shape, nkx, nky, nkz, sx, sy, sz):
    random_generator = RandomGenerator(seed=0)

    # generate random image
    image = random_generator.complex64_tensor(size=im_shape)

    # create random trajectories
    k_list = []
    for spacing, nk in zip([sz, sy, sx], [nkz, nky, nkx]):
        if spacing == 'nuf':
            k = random_generator.float32_tensor(size=nk)
        elif spacing == 'uf':
            k = create_uniform_traj(nk)
        elif spacing == 'z':
            k = torch.zeros(nk)
        k_list.append(k)
    ktraj = KTrajectory(*k_list)

    return image, ktraj


@pytest.mark.parametrize(
    'im_shape, k_shape, nkx, nky, nkz, sx, sy, sz',
    [
        # 2d cart mri with 1 coil, no oversampling
        (
            (1, 1, 1, 96, 128),  # im shape
            (1, 1, 1, 96, 128),  # k shape
            (1, 1, 1, 128),  # kx
            (1, 1, 96, 1),  # ky
            (1, 1, 1, 1),  # kz
            'uf',  # kx is uniform
            'uf',  # ky is uniform
            'z',  # zero so no Fourier transform is performed along that dimension
        ),
        # 2d cart mri with 1 coil, with oversampling
        (
            (1, 1, 1, 96, 128),
            (1, 1, 1, 128, 192),
            (1, 1, 1, 192),
            (1, 1, 128, 1),
            (1, 1, 1, 1),
            'uf',
            'uf',
            'z',
        ),
        # 2d non-Cartesian mri with 2 coil
        (
            (1, 2, 1, 96, 128),
            (1, 2, 1, 16, 192),
            (1, 1, 16, 192),
            (1, 1, 16, 192),
            (1, 1, 1, 1),
            'nuf',  # kx is non-uniform
            'nuf',
            'z',
        ),
        # 3d nuFFT mri, 4 coils, 2 other
        (
            (2, 4, 16, 32, 64),
            (2, 4, 16, 32, 64),
            (2, 16, 32, 64),
            (2, 16, 32, 64),
            (2, 16, 32, 64),
            'nuf',
            'nuf',
            'nuf',
        ),
        # 2d nuFFT cine mri with 8 cardiac phases, 5 coils
        (
            (8, 5, 1, 64, 64),
            (8, 5, 1, 18, 128),
            (8, 1, 18, 128),
            (8, 1, 18, 128),
            (8, 1, 1, 1),
            'nuf',
            'nuf',
            'z',
        ),
        # 2d cart cine mri with 9 cardiac phases, 6 coils
        (
            (9, 6, 1, 96, 128),
            (9, 6, 1, 128, 192),
            (9, 1, 1, 192),
            (9, 1, 128, 1),
            (9, 1, 1, 1),
            'uf',
            'uf',
            'z',
        ),
        # 2d cart cine mri with 8 cardiac phases, 7 coils, with oversampling
        (
            (8, 7, 1, 64, 96),
            (8, 7, 1, 96, 128),
            (8, 1, 1, 128),
            (8, 1, 96, 1),
            (8, 1, 1, 1),
            'uf',
            'uf',
            'z',
        ),
        # radial phase encoding (RPE), 8 coils, with oversampling in both FFT and nuFFT directions
        (
            (2, 8, 64, 32, 48),
            (2, 8, 8, 64, 96),
            (2, 1, 1, 96),
            (2, 8, 64, 1),
            (2, 8, 64, 1),
            'uf',
            'nuf',
            'nuf',
        ),
        # stack of stars, 5 other, 3 coil, oversampling in both FFT and nuFFT directions
        (
            (5, 3, 48, 16, 32),
            (5, 3, 96, 18, 64),
            (5, 1, 18, 64),
            (5, 1, 18, 64),
            (5, 96, 1, 1),
            'nuf',
            'nuf',
            'uf',
        ),
        # similar to above, but the nuFFT dimensions are not next to each other
        (
            (5, 3, 48, 16, 32),
            (5, 3, 96, 18, 64),
            (5, 1, 18, 64),
            (5, 96, 1, 1),
            (5, 1, 18, 64),
            'nuf',
            'uf',
            'nuf',
        ),
    ],
)
def test_fourier_fwd_adj_property(im_shape, k_shape, nkx, nky, nkz, sx, sy, sz):
    """Test adjoint property of Fourier operator."""

    # generate random images and k-space trajectories
    image, ktraj = create_data(im_shape, nkx, nky, nkz, sx, sy, sz)

    # create operator
    recon_shape = SpatialDimension(im_shape[-3], im_shape[-2], im_shape[-1])
    encoding_shape = SpatialDimension(k_shape[-3], k_shape[-2], k_shape[-1])
    op = FourierOp(recon_shape=recon_shape, encoding_shape=encoding_shape, traj=ktraj)

    # apply forward and adjoint operator
    kdata = op(image)
    reco = op.H(kdata)

    # test adjoint property; i.e. <Fu,v> == <u, F^Hv> for all u,v
    random_generator = RandomGenerator(seed=0)
    u = random_generator.complex64_tensor(size=image.shape)
    v = random_generator.complex64_tensor(size=kdata.shape)
    Fu, FHv = op(u), op.H(v)

    Fu_v = torch.vdot(Fu.flatten(), v.flatten())
    u_FHv = torch.vdot(u.flatten(), FHv.flatten())

    assert reco.shape == image.shape
    assert torch.isclose(Fu_v, u_FHv, rtol=1e-3)
