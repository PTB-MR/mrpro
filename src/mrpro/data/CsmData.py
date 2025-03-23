"""Class for coil sensitivity maps (csm)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from mrpro.algorithms.prewhiten_kspace import prewhiten_kspace
from mrpro.data.DcfData import DcfData
from mrpro.data.IData import IData
from mrpro.data.QData import QData
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.operators.FourierOp import FourierOp

if TYPE_CHECKING:
    from mrpro.data.KData import KData
    from mrpro.data.KNoise import KNoise
    from mrpro.operators.SensitivityOp import SensitivityOp


class CsmData(QData):
    """Coil sensitivity map class."""

    @staticmethod
    def _reconstruct_coil_images(kdata: KData, noise: KNoise | None = None) -> IData:
        """Direct reconstruction applying density compensation and adjoint of Fourier operator.

        Parameters
        ----------
        kdata
            k-space data
        noise, optional
            Noise measurement for prewhitening.

        Returns
        -------
            reconstructed coil-wise images
        """
        if noise is not None:
            kdata = prewhiten_kspace(kdata, noise)
        fourier_op = FourierOp.from_kdata(kdata)
        dcf = DcfData.from_traj_voronoi(kdata.traj)
        adjoint_operator_with_dcf = fourier_op.H @ dcf.as_operator()
        (img_tensor,) = adjoint_operator_with_dcf(kdata.data)
        return IData.from_tensor_and_kheader(img_tensor, kdata.header)

    @classmethod
    def from_kdata_walsh(
        cls,
        kdata: KData,
        noise: KNoise | None = None,
        smoothing_width: int | SpatialDimension[int] = 5,
        chunk_size_otherdim: int | None = None,
    ) -> Self:
        """Create csm object from k-space data using Walsh method.

        See also `~mrpro.algorithms.csm.walsh`.

        Parameters
        ----------
        kdata
            k-space data
        noise, optional
            Noise measurement for prewhitening.
        smoothing_width
            width of smoothing filter
        chunk_size_otherdim
            How many elements of the other dimensions should be processed at once.
            Default is `None`, which means that all elements are processed at once.

        Returns
        -------
            Coil sensitivity maps
        """
        return cls.from_idata_walsh(cls._reconstruct_coil_images(kdata, noise), smoothing_width, chunk_size_otherdim)

    @classmethod
    def from_idata_walsh(
        cls,
        idata: IData,
        smoothing_width: int | SpatialDimension[int] = 5,
        chunk_size_otherdim: int | None = None,
    ) -> Self:
        """Create csm object from image data using Walsh method.

        See also `~mrpro.algorithms.csm.walsh`.

        Parameters
        ----------
        idata
            IData object containing the images for each coil element.
        smoothing_width
            width of smoothing filter.
        chunk_size_otherdim:
            How many elements of the other dimensions should be processed at once.
            Default is `None`, which means that all elements are processed at once.

        Returns
        -------
            Coil sensitivity maps
        """
        from mrpro.algorithms.csm.walsh import walsh

        # convert smoothing_width to SpatialDimension if int
        if isinstance(smoothing_width, int):
            smoothing_width = SpatialDimension(smoothing_width, smoothing_width, smoothing_width)

        csm_fun = torch.vmap(
            lambda img: walsh(img, smoothing_width),
            chunk_size=chunk_size_otherdim,
        )
        csm_tensor = csm_fun(idata.data.flatten(end_dim=-5)).reshape(idata.data.shape)
        csm = cls(header=idata.header, data=csm_tensor)
        return csm

    @classmethod
    def from_idata_inati(
        cls,
        idata: IData,
        smoothing_width: int | SpatialDimension[int] = 5,
        chunk_size_otherdim: int | None = None,
    ) -> Self:
        """Create csm object from image data using Inati method.

        Parameters
        ----------
        idata
            IData object containing the images for each coil element.
        smoothing_width
            Size of the smoothing kernel.
        chunk_size_otherdim:
            How many elements of the other dimensions should be processed at once.
            Default is None, which means that all elements are processed at once.

        Returns
        -------
            Coil sensitivity maps
        """
        from mrpro.algorithms.csm.inati import inati

        csm_fun = torch.vmap(lambda img: inati(img, smoothing_width), chunk_size=chunk_size_otherdim)
        csm_tensor = csm_fun(idata.data)
        csm = cls(header=idata.header, data=csm_tensor)
        return csm

    def as_operator(self) -> SensitivityOp:
        """Create SensitivityOp using a copy of the CSMs."""
        from mrpro.operators.SensitivityOp import SensitivityOp

        return SensitivityOp(self)
