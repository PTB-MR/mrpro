"""Tests for TotalVariationRegularizedReconstruction."""

import pytest
from mr2.algorithms.reconstruction import TotalVariationRegularizedReconstruction
from mr2.data import CsmData, DcfData, KData
from mr2.operators import FourierOp


def test_total_variation_automatic(cartesian_kdata: KData) -> None:
    """Test automatic setup from kdata only."""
    reconstruction = TotalVariationRegularizedReconstruction(
        kdata=cartesian_kdata,
        regularization_dim=(-1, -2),
        regularization_weight=0.01,
        max_iterations=2,
    )
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.csm_op is not None
    assert reconstruction.dcf_op is not None


def test_total_variation_with_callable_csm(cartesian_kdata: KData) -> None:
    """Test with callable CSM estimation."""
    reconstruction = TotalVariationRegularizedReconstruction(
        kdata=cartesian_kdata,
        csm=CsmData.from_idata_walsh,
        regularization_dim=(-1, -2),
        regularization_weight=0.01,
        max_iterations=2,
    )
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.csm_op is not None


def test_total_variation_with_explicit_csm(cartesian_kdata: KData) -> None:
    """Test with pre-computed CSM."""
    csm = CsmData.from_kdata_walsh(cartesian_kdata)
    reconstruction = TotalVariationRegularizedReconstruction(
        kdata=cartesian_kdata,
        csm=csm,
        regularization_dim=(-1, -2),
        regularization_weight=0.01,
        max_iterations=2,
    )
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.csm_op is not None


def test_total_variation_with_explicit_dcf(cartesian_kdata: KData) -> None:
    """Test with pre-computed DCF."""
    dcf = DcfData.from_traj_voronoi(cartesian_kdata.traj)
    reconstruction = TotalVariationRegularizedReconstruction(
        kdata=cartesian_kdata,
        dcf=dcf,
        regularization_dim=(-1, -2),
        regularization_weight=0.01,
        max_iterations=2,
    )
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.dcf_op is not None


@pytest.mark.cuda
def test_total_variation_cuda_from_kdata(cartesian_kdata: KData) -> None:
    """Test CUDA device transfers for reconstruction created from kdata."""
    reconstruction = TotalVariationRegularizedReconstruction(
        kdata=cartesian_kdata,
        regularization_dim=(-1, -2),
        regularization_weight=0.01,
        max_iterations=2,
    ).cuda()
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    reconstruction = TotalVariationRegularizedReconstruction(
        kdata=cartesian_kdata.cuda(),
        regularization_dim=(-1, -2),
        regularization_weight=0.01,
        max_iterations=2,
    )
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    reconstruction = TotalVariationRegularizedReconstruction(
        kdata=cartesian_kdata.cuda(),
        regularization_dim=(-1, -2),
        regularization_weight=0.01,
        max_iterations=2,
    ).cpu()
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cpu


@pytest.mark.cuda
def test_total_variation_cuda_explicit_components(cartesian_kdata: KData) -> None:
    """Test CUDA device transfers with explicit FourierOp, CSM, and DCF."""

    def explicit_components(kdata: KData) -> tuple[FourierOp, CsmData, DcfData]:
        fourier_op = FourierOp.from_kdata(kdata)
        csm = CsmData.from_kdata_walsh(kdata)
        dcf = DcfData.from_traj_voronoi(kdata.traj)
        return fourier_op, csm, dcf

    fourier_op, csm, dcf = explicit_components(cartesian_kdata)
    reconstruction = TotalVariationRegularizedReconstruction(
        fourier_op=fourier_op,
        csm=csm,
        dcf=dcf,
        regularization_dim=(-1, -2),
        regularization_weight=0.01,
        max_iterations=2,
    ).cuda()
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    fourier_op, csm, dcf = explicit_components(cartesian_kdata.cuda())
    reconstruction = TotalVariationRegularizedReconstruction(
        fourier_op=fourier_op,
        csm=csm,
        dcf=dcf,
        regularization_dim=(-1, -2),
        regularization_weight=0.01,
        max_iterations=2,
    )
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    fourier_op, csm, dcf = explicit_components(cartesian_kdata.cuda())
    reconstruction = TotalVariationRegularizedReconstruction(
        fourier_op=fourier_op,
        csm=csm,
        dcf=dcf,
        regularization_dim=(-1, -2),
        regularization_weight=0.01,
        max_iterations=2,
    ).cpu()
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cpu
