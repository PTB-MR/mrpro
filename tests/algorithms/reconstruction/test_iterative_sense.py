"""Tests for IterativeSENSEReconstruction."""

import pytest
from mr2.algorithms.reconstruction import IterativeSENSEReconstruction
from mr2.data import CsmData, DcfData, KData
from mr2.operators import FourierOp


def test_iterative_sense_automatic(cartesian_kdata: KData) -> None:
    """Test automatic setup from kdata only."""
    reconstruction = IterativeSENSEReconstruction(kdata=cartesian_kdata, n_iterations=2)
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.csm_op is not None
    assert reconstruction.dcf_op is not None


def test_iterative_sense_with_callable_csm(cartesian_kdata: KData) -> None:
    """Test with callable CSM estimation."""
    reconstruction = IterativeSENSEReconstruction(
        kdata=cartesian_kdata,
        csm=CsmData.from_idata_walsh,
        n_iterations=2,
    )
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.csm_op is not None


def test_iterative_sense_with_explicit_csm(cartesian_kdata: KData) -> None:
    """Test with pre-computed CSM."""
    csm = CsmData.from_kdata_walsh(cartesian_kdata)
    reconstruction = IterativeSENSEReconstruction(kdata=cartesian_kdata, csm=csm, n_iterations=2)
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.csm_op is not None


def test_iterative_sense_with_explicit_dcf(cartesian_kdata: KData) -> None:
    """Test with pre-computed DCF."""
    dcf = DcfData.from_traj_voronoi(cartesian_kdata.traj)
    reconstruction = IterativeSENSEReconstruction(kdata=cartesian_kdata, dcf=dcf, n_iterations=2)
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert reconstruction.dcf_op is not None


@pytest.mark.cuda
def test_iterative_sense_cuda_from_kdata(cartesian_kdata: KData) -> None:
    """Test CUDA device transfers for reconstruction created from kdata."""
    reconstruction = IterativeSENSEReconstruction(kdata=cartesian_kdata, n_iterations=2).cuda()
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    reconstruction = IterativeSENSEReconstruction(kdata=cartesian_kdata.cuda(), n_iterations=2)
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    reconstruction = IterativeSENSEReconstruction(kdata=cartesian_kdata.cuda(), n_iterations=2).cpu()
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cpu


@pytest.mark.cuda
def test_iterative_sense_cuda_explicit_components(cartesian_kdata: KData) -> None:
    """Test CUDA device transfers with explicit FourierOp, CSM, and DCF."""

    def explicit_components(kdata: KData) -> tuple[FourierOp, CsmData, DcfData]:
        fourier_op = FourierOp.from_kdata(kdata)
        csm = CsmData.from_kdata_walsh(kdata)
        dcf = DcfData.from_traj_voronoi(kdata.traj)
        return fourier_op, csm, dcf

    fourier_op, csm, dcf = explicit_components(cartesian_kdata)
    reconstruction = IterativeSENSEReconstruction(fourier_op=fourier_op, csm=csm, dcf=dcf, n_iterations=2).cuda()
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    fourier_op, csm, dcf = explicit_components(cartesian_kdata.cuda())
    reconstruction = IterativeSENSEReconstruction(fourier_op=fourier_op, csm=csm, dcf=dcf, n_iterations=2)
    idata = reconstruction(cartesian_kdata.cuda())
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cuda

    fourier_op, csm, dcf = explicit_components(cartesian_kdata.cuda())
    reconstruction = IterativeSENSEReconstruction(fourier_op=fourier_op, csm=csm, dcf=dcf, n_iterations=2).cpu()
    idata = reconstruction(cartesian_kdata)
    assert idata.data.shape[-3:] == cartesian_kdata.header.recon_matrix.zyx
    assert idata.is_cpu
