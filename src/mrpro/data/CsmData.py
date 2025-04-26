"""Class for coil sensitivity maps (csm)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from mrpro.data.IData import IData
from mrpro.data.QData import QData
from mrpro.data.QHeader import QHeader
from mrpro.data.SpatialDimension import SpatialDimension

if TYPE_CHECKING:
    from mrpro.data.KData import KData
    from mrpro.data.KNoise import KNoise
    from mrpro.operators.SensitivityOp import SensitivityOp


class CsmData(QData):
    """Coil sensitivity map class."""

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
        from mrpro.algorithms.reconstruction import DirectReconstruction

        return cls.from_idata_walsh(
            DirectReconstruction(kdata, noise=noise, csm=None)(kdata), smoothing_width, chunk_size_otherdim
        )

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
        csm = cls(header=QHeader.from_iheader(idata.header), data=csm_tensor)
        return csm

    @classmethod
    def from_kdata_inati(
        cls,
        kdata: KData,
        noise: KNoise | None = None,
        smoothing_width: int | SpatialDimension[int] = 5,
        chunk_size_otherdim: int | None = None,
    ) -> Self:
        """Create csm object from k-space data using Inati method.

        See also `~mrpro.algorithms.csm.inati`.

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
        from mrpro.algorithms.reconstruction import DirectReconstruction

        return cls.from_idata_inati(
            DirectReconstruction(kdata, noise=noise, csm=None)(kdata), smoothing_width, chunk_size_otherdim
        )

    @classmethod
    def from_idata_inati(
        cls,
        idata: IData,
        smoothing_width: int | SpatialDimension[int] = 5,
        chunk_size_otherdim: int | None = None,
    ) -> Self:
        """Create csm object from image data using Inati method.

        See also `~mrpro.algorithms.csm.inati`.

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
        csm = cls(header=QHeader.from_iheader(idata.header), data=csm_tensor)
        return csm

    def as_operator(self) -> SensitivityOp:
        """Create SensitivityOp using a copy of the CSMs."""
        from mrpro.operators.SensitivityOp import SensitivityOp

        return SensitivityOp(self)
