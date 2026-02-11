"""Tests for DirectReconstruction."""

import pytest
from mrpro.algorithms.reconstruction import DirectReconstruction
from mrpro.data import CsmData, KData
from mrpro.operators import FourierOp


def test_direct_reconstruction_automatic(cartesian_kdata: KData) -> None:
    """Test automatic setup from kdata only."""
    reconstruction = DirectReconstruction(kdata=cartesian_kdata)
    idata = reconstruction(cartesian_kdata)
    recon = cartesian_kdata.header.recon_matrix
    assert idata.data.shape[-3:] == (recon.z, recon.y, recon.x)
    assert reconstruction.csm is not None
    assert reconstruction.dcf is not None


def test_direct_reconstruction_with_explicit_csm(cartesian_kdata: KData) -> None:
    """Test with pre-computed CSM."""
    csm = CsmData.from_idata_walsh(DirectReconstruction(kdata=cartesian_kdata)(cartesian_kdata))
    reconstruction = DirectReconstruction(kdata=cartesian_kdata, csm=csm)
    idata = reconstruction(cartesian_kdata)
    recon = cartesian_kdata.header.recon_matrix
    assert idata.data.shape[-3:] == (recon.z, recon.y, recon.x)
    assert reconstruction.csm is csm


def test_direct_reconstruction_with_explicit_dcf(cartesian_kdata: KData) -> None:
    """Test with pre-computed DCF."""
    dcf = DirectReconstruction(kdata=cartesian_kdata).dcf
    reconstruction = DirectReconstruction(kdata=cartesian_kdata, dcf=dcf)
    idata = reconstruction(cartesian_kdata)
    recon = cartesian_kdata.header.recon_matrix
    assert idata.data.shape[-3:] == (recon.z, recon.y, recon.x)
    assert reconstruction.dcf is dcf


def test_direct_reconstruction_with_explicit_fourier_op(cartesian_kdata: KData) -> None:
    """Test with pre-computed FourierOp."""
    fourier_op = FourierOp.from_kdata(cartesian_kdata)
    reconstruction = DirectReconstruction(kdata=cartesian_kdata, fourier_op=fourier_op)
    idata = reconstruction(cartesian_kdata)
    recon = cartesian_kdata.header.recon_matrix
    assert idata.data.shape[-3:] == (recon.z, recon.y, recon.x)
    assert reconstruction.fourier_op is fourier_op


def test_direct_reconstruction_csm_false_skips_coil_combination(cartesian_kdata: KData) -> None:
    """Test that csm=False in direct_reconstruction skips coil combination."""
    reconstruction = DirectReconstruction(kdata=cartesian_kdata, csm=CsmData.from_idata_walsh)
    idata_with_csm = reconstruction.direct_reconstruction(cartesian_kdata)
    idata_without_csm = reconstruction.direct_reconstruction(cartesian_kdata, csm=False)
    assert idata_with_csm.data.shape[-4] == 1
    assert idata_without_csm.data.shape[-4] == cartesian_kdata.data.shape[-4]


@pytest.mark.cuda
def test_direct_reconstruction_cuda(cartesian_kdata: KData) -> None:
    """Test on CUDA device."""
    reconstruction = DirectReconstruction(kdata=cartesian_kdata).cuda()
    idata = reconstruction(cartesian_kdata.cuda())
    recon = cartesian_kdata.header.recon_matrix
    assert idata.data.shape[-3:] == (recon.z, recon.y, recon.x)
    assert idata.is_cuda
