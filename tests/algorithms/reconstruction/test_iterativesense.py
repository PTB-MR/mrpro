"""Tests for IterativeSENSEReconstruction."""

import pytest
from mrpro.algorithms.reconstruction import IterativeSENSEReconstruction
from mrpro.data import CsmData, KData


def test_iterative_sense_automatic(cartesian_kdata: KData) -> None:
    """Test automatic setup from kdata only."""
    reconstruction = IterativeSENSEReconstruction(kdata=cartesian_kdata, n_iterations=2)
    idata = reconstruction(cartesian_kdata)
    recon = cartesian_kdata.header.recon_matrix
    assert idata.data.shape[-3:] == (recon.z, recon.y, recon.x)
    assert reconstruction.csm is not None
    assert reconstruction.dcf is not None


def test_iterative_sense_with_callable_csm(cartesian_kdata: KData) -> None:
    """Test with callable CSM estimation."""
    reconstruction = IterativeSENSEReconstruction(
        kdata=cartesian_kdata,
        csm=CsmData.from_idata_walsh,
        n_iterations=2,
    )
    idata = reconstruction(cartesian_kdata)
    recon = cartesian_kdata.header.recon_matrix
    assert idata.data.shape[-3:] == (recon.z, recon.y, recon.x)
    assert reconstruction.csm is not None


def test_iterative_sense_with_explicit_csm(cartesian_kdata: KData) -> None:
    """Test with pre-computed CSM."""
    csm = CsmData.from_idata_walsh(
        IterativeSENSEReconstruction(kdata=cartesian_kdata, n_iterations=2)(cartesian_kdata),
    )
    reconstruction = IterativeSENSEReconstruction(kdata=cartesian_kdata, csm=csm, n_iterations=2)
    idata = reconstruction(cartesian_kdata)
    recon = cartesian_kdata.header.recon_matrix
    assert idata.data.shape[-3:] == (recon.z, recon.y, recon.x)
    assert reconstruction.csm is csm


def test_iterative_sense_with_explicit_dcf(cartesian_kdata: KData) -> None:
    """Test with pre-computed DCF."""
    dcf = IterativeSENSEReconstruction(kdata=cartesian_kdata, n_iterations=2).dcf
    reconstruction = IterativeSENSEReconstruction(kdata=cartesian_kdata, dcf=dcf, n_iterations=2)
    idata = reconstruction(cartesian_kdata)
    recon = cartesian_kdata.header.recon_matrix
    assert idata.data.shape[-3:] == (recon.z, recon.y, recon.x)
    assert reconstruction.dcf is dcf


@pytest.mark.cuda
def test_iterative_sense_cuda(cartesian_kdata: KData) -> None:
    """Test on CUDA device."""
    reconstruction = IterativeSENSEReconstruction(kdata=cartesian_kdata, n_iterations=2).cuda()
    idata = reconstruction(cartesian_kdata.cuda())
    recon = cartesian_kdata.header.recon_matrix
    assert idata.data.shape[-3:] == (recon.z, recon.y, recon.x)
    assert idata.is_cuda
