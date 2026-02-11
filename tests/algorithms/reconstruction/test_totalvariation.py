"""Tests for TotalVariationRegularizedReconstruction."""

import pytest
from mrpro.algorithms.reconstruction import TotalVariationRegularizedReconstruction
from mrpro.data import CsmData, KData


def test_total_variation_automatic(cartesian_kdata: KData) -> None:
    """Test automatic setup from kdata only."""
    reconstruction = TotalVariationRegularizedReconstruction(
        kdata=cartesian_kdata,
        regularization_dim=(-1, -2),
        regularization_weight=0.01,
        max_iterations=2,
    )
    idata = reconstruction(cartesian_kdata)
    recon = cartesian_kdata.header.recon_matrix
    assert idata.data.shape[-3:] == (recon.z, recon.y, recon.x)
    assert reconstruction.csm is not None
    assert reconstruction.dcf is not None


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
    recon = cartesian_kdata.header.recon_matrix
    assert idata.data.shape[-3:] == (recon.z, recon.y, recon.x)
    assert reconstruction.csm is not None


def test_total_variation_with_explicit_csm(cartesian_kdata: KData) -> None:
    """Test with pre-computed CSM."""
    csm = CsmData.from_idata_walsh(
        TotalVariationRegularizedReconstruction(
            kdata=cartesian_kdata,
            regularization_dim=(-1, -2),
            regularization_weight=0.01,
            max_iterations=2,
        )(cartesian_kdata),
    )
    reconstruction = TotalVariationRegularizedReconstruction(
        kdata=cartesian_kdata,
        csm=csm,
        regularization_dim=(-1, -2),
        regularization_weight=0.01,
        max_iterations=2,
    )
    idata = reconstruction(cartesian_kdata)
    recon = cartesian_kdata.header.recon_matrix
    assert idata.data.shape[-3:] == (recon.z, recon.y, recon.x)
    assert reconstruction.csm is csm


def test_total_variation_with_explicit_dcf(cartesian_kdata: KData) -> None:
    """Test with pre-computed DCF."""
    dcf = TotalVariationRegularizedReconstruction(
        kdata=cartesian_kdata,
        regularization_dim=(-1, -2),
        regularization_weight=0.01,
        max_iterations=2,
    ).dcf
    reconstruction = TotalVariationRegularizedReconstruction(
        kdata=cartesian_kdata,
        dcf=dcf,
        regularization_dim=(-1, -2),
        regularization_weight=0.01,
        max_iterations=2,
    )
    idata = reconstruction(cartesian_kdata)
    recon = cartesian_kdata.header.recon_matrix
    assert idata.data.shape[-3:] == (recon.z, recon.y, recon.x)
    assert reconstruction.dcf is dcf


@pytest.mark.cuda
def test_total_variation_cuda(cartesian_kdata: KData) -> None:
    """Test on CUDA device."""
    reconstruction = TotalVariationRegularizedReconstruction(
        kdata=cartesian_kdata,
        regularization_dim=(-1, -2),
        regularization_weight=0.01,
        max_iterations=2,
    ).cuda()
    idata = reconstruction(cartesian_kdata.cuda())
    recon = cartesian_kdata.header.recon_matrix
    assert idata.data.shape[-3:] == (recon.z, recon.y, recon.x)
    assert idata.is_cuda
