"""Tests for PlugAndPlayPriorsReconstruction."""

import pytest
import torch
from mrpro.algorithms import total_variation_denoising
from mrpro.algorithms.reconstruction import PlugAndPlayPriorsReconstruction
from mrpro.data import CsmData, DcfData, KData
from mrpro.operators import DensityCompensationOp, FourierOp


def tv_denoiser(image: torch.Tensor) -> torch.Tensor:
    """TV denoiser used as PnP prior."""
    return total_variation_denoising(image, regularization_dim=(-1, -2), regularization_weight=0.01, max_iterations=2)


def test_plug_and_play_priors_automatic(cartesian_kdata: KData) -> None:
    """Test automatic setup from kdata only."""
    reconstruction = PlugAndPlayPriorsReconstruction(
        kdata=cartesian_kdata,
        denoiser=tv_denoiser,
        admm_regularization_strength=0.1,
        max_iterations=2,
        max_iterations_cg=2,
    )
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.csm_op is not None
    assert reconstruction.dcf_op is not None


def test_plug_and_play_priors_with_callable_csm(cartesian_kdata: KData) -> None:
    """Test with callable CSM estimation."""
    reconstruction = PlugAndPlayPriorsReconstruction(
        kdata=cartesian_kdata,
        csm=CsmData.from_idata_walsh,
        denoiser=tv_denoiser,
        admm_regularization_strength=0.1,
        max_iterations=2,
        max_iterations_cg=2,
    )
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.csm_op is not None


def test_plug_and_play_priors_with_explicit_csm(cartesian_kdata: KData) -> None:
    """Test with pre-computed CSM."""
    csm = CsmData.from_kdata_walsh(cartesian_kdata)
    reconstruction = PlugAndPlayPriorsReconstruction(
        kdata=cartesian_kdata,
        csm=csm,
        denoiser=tv_denoiser,
        admm_regularization_strength=0.1,
        max_iterations=2,
        max_iterations_cg=2,
    )
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.csm_op is not None
    torch.testing.assert_close(reconstruction.csm_op.csm_tensor, csm.data)


def test_plug_and_play_priors_with_explicit_dcf(cartesian_kdata: KData) -> None:
    """Test with pre-computed DCF."""
    dcf = DcfData.from_traj_voronoi(cartesian_kdata.traj)
    reconstruction = PlugAndPlayPriorsReconstruction(
        kdata=cartesian_kdata,
        dcf=dcf,
        denoiser=tv_denoiser,
        admm_regularization_strength=0.1,
        max_iterations=2,
        max_iterations_cg=2,
    )
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert isinstance(reconstruction.dcf_op, DensityCompensationOp)


def test_plug_and_play_priors_denoiser_calls(cartesian_kdata: KData) -> None:
    """Test that the denoiser is called once per ADMM iteration with an image-shaped tensor."""
    calls: list[torch.Size] = []

    def counting_denoiser(image: torch.Tensor) -> torch.Tensor:
        calls.append(image.shape)
        return image

    reconstruction = PlugAndPlayPriorsReconstruction(
        kdata=cartesian_kdata,
        denoiser=counting_denoiser,
        admm_regularization_strength=0.1,
        max_iterations=3,
        max_iterations_cg=2,
    )
    idata = reconstruction(cartesian_kdata)
    assert len(calls) == 3
    assert all(shape == idata.data.shape for shape in calls)


def test_plug_and_play_priors_tolerance(cartesian_kdata: KData) -> None:
    """Test that a large tolerance stops the iterations early."""
    n_calls = 0

    def counting_denoiser(image: torch.Tensor) -> torch.Tensor:
        nonlocal n_calls
        n_calls += 1
        return image

    reconstruction = PlugAndPlayPriorsReconstruction(
        kdata=cartesian_kdata,
        denoiser=counting_denoiser,
        admm_regularization_strength=0.1,
        max_iterations=10,
        max_iterations_cg=2,
        tolerance=1e10,
    )
    reconstruction(cartesian_kdata)
    assert n_calls == 1


@pytest.mark.cuda
def test_plug_and_play_priors_cuda_from_kdata(cartesian_kdata: KData) -> None:
    """Test CUDA device transfers for reconstruction created from kdata."""
    reconstruction = PlugAndPlayPriorsReconstruction(
        kdata=cartesian_kdata,
        denoiser=tv_denoiser,
        admm_regularization_strength=0.1,
        max_iterations=2,
        max_iterations_cg=2,
    ).cuda()
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    reconstruction = PlugAndPlayPriorsReconstruction(
        kdata=cartesian_kdata.cuda(),
        denoiser=tv_denoiser,
        admm_regularization_strength=0.1,
        max_iterations=2,
        max_iterations_cg=2,
    )
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    reconstruction = PlugAndPlayPriorsReconstruction(
        kdata=cartesian_kdata.cuda(),
        denoiser=tv_denoiser,
        admm_regularization_strength=0.1,
        max_iterations=2,
        max_iterations_cg=2,
    ).cpu()
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cpu


@pytest.mark.cuda
def test_plug_and_play_priors_cuda_explicit_components(cartesian_kdata: KData) -> None:
    """Test CUDA device transfers with explicit FourierOp, CSM, and DCF."""

    def explicit_components(kdata: KData) -> tuple[FourierOp, CsmData, DcfData]:
        fourier_op = FourierOp.from_kdata(kdata)
        csm = CsmData.from_kdata_walsh(kdata)
        dcf = DcfData.from_traj_voronoi(kdata.traj)
        return fourier_op, csm, dcf

    fourier_op, csm, dcf = explicit_components(cartesian_kdata)
    reconstruction = PlugAndPlayPriorsReconstruction(
        fourier_op=fourier_op,
        csm=csm,
        dcf=dcf,
        denoiser=tv_denoiser,
        admm_regularization_strength=0.1,
        max_iterations=2,
        max_iterations_cg=2,
    ).cuda()
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    fourier_op, csm, dcf = explicit_components(cartesian_kdata.cuda())
    reconstruction = PlugAndPlayPriorsReconstruction(
        fourier_op=fourier_op,
        csm=csm,
        dcf=dcf,
        denoiser=tv_denoiser,
        admm_regularization_strength=0.1,
        max_iterations=2,
        max_iterations_cg=2,
    )
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    fourier_op, csm, dcf = explicit_components(cartesian_kdata.cuda())
    reconstruction = PlugAndPlayPriorsReconstruction(
        fourier_op=fourier_op,
        csm=csm,
        dcf=dcf,
        denoiser=tv_denoiser,
        admm_regularization_strength=0.1,
        max_iterations=2,
        max_iterations_cg=2,
    ).cpu()
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cpu
