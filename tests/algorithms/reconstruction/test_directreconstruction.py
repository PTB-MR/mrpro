"""Tests for DirectReconstruction."""

from collections.abc import Callable

import pytest
from mr2.algorithms.reconstruction import DirectReconstruction
from mr2.data import CsmData, DcfData, KData
from mr2.operators import DensityCompensationOp, FourierOp, SensitivityOp


def test_direct_reconstruction_automatic(cartesian_kdata: KData) -> None:
    """Test automatic setup from kdata only."""
    reconstruction = DirectReconstruction(kdata=cartesian_kdata)
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.csm_op is not None
    assert reconstruction.dcf_op is not None


def test_direct_reconstruction_with_explicit_csm(cartesian_kdata: KData) -> None:
    """Test with pre-computed CSM."""
    csm = CsmData.from_idata_walsh(DirectReconstruction(kdata=cartesian_kdata)(cartesian_kdata))
    reconstruction = DirectReconstruction(kdata=cartesian_kdata, csm=csm)
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert isinstance(reconstruction.csm_op, SensitivityOp)


def test_direct_reconstruction_with_explicit_csm_op(cartesian_kdata: KData) -> None:
    """Test with pre-computed CSM operator."""
    csm = CsmData.from_idata_walsh(DirectReconstruction(kdata=cartesian_kdata)(cartesian_kdata))
    csm_op = csm.as_operator()
    reconstruction = DirectReconstruction(kdata=cartesian_kdata, csm=csm_op)
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.csm_op is csm_op


def test_direct_reconstruction_with_explicit_dcf(cartesian_kdata: KData) -> None:
    """Test with pre-computed DCF."""
    dcf = DcfData.from_traj_voronoi(cartesian_kdata.traj)
    reconstruction = DirectReconstruction(kdata=cartesian_kdata, dcf=dcf)
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert isinstance(reconstruction.dcf_op, DensityCompensationOp)


def test_direct_reconstruction_with_explicit_dcf_op(cartesian_kdata: KData) -> None:
    """Test with pre-computed DCF operator."""
    dcf_op = DcfData.from_traj_voronoi(cartesian_kdata.traj).as_operator()
    reconstruction = DirectReconstruction(kdata=cartesian_kdata, dcf=dcf_op)
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.dcf_op is dcf_op


def test_direct_reconstruction_with_explicit_fourier_op(cartesian_kdata: KData) -> None:
    """Test with pre-computed FourierOp."""
    fourier_op = FourierOp.from_kdata(cartesian_kdata)
    reconstruction = DirectReconstruction(kdata=cartesian_kdata, fourier_op=fourier_op)
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.fourier_op is fourier_op


@pytest.mark.cuda
def test_direct_reconstruction_cuda_from_kdata(cartesian_kdata: KData) -> None:
    """Test CUDA device transfers for reconstruction created from kdata."""
    reconstruction = DirectReconstruction(kdata=cartesian_kdata).cuda()
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    reconstruction = DirectReconstruction(kdata=cartesian_kdata.cuda())
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    reconstruction = DirectReconstruction(kdata=cartesian_kdata.cuda()).cpu()
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cpu


@pytest.mark.cuda
def test_direct_reconstruction_cuda_explicit_components(
    cartesian_kdata: KData,
    explicit_components: Callable[[KData], tuple[FourierOp, CsmData, DcfData]],
) -> None:
    """Test CUDA device transfers with explicit FourierOp, CSM, and DCF."""
    fourier_op, csm, dcf = explicit_components(cartesian_kdata)
    reconstruction = DirectReconstruction(fourier_op=fourier_op, csm=csm, dcf=dcf).cuda()
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    fourier_op, csm, dcf = explicit_components(cartesian_kdata.cuda())
    reconstruction = DirectReconstruction(fourier_op=fourier_op, csm=csm, dcf=dcf)
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    fourier_op, csm, dcf = explicit_components(cartesian_kdata.cuda())
    reconstruction = DirectReconstruction(fourier_op=fourier_op, csm=csm, dcf=dcf).cpu()
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cpu
