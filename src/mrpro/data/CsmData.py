"""Class for coil sensitivity maps (csm)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from mrpro.data.IData import IData
from mrpro.data.QData import QData
from mrpro.data.QHeader import QHeader
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.interpolate import apply_lowres

if TYPE_CHECKING:
    from mrpro.data.KData import KData
    from mrpro.data.KNoise import KNoise
    from mrpro.operators.SensitivityOp import SensitivityOp


def get_downsampled_size(
    data_size: torch.Size, downsampled_size: int | SpatialDimension[int] | None
) -> tuple[int, int, int]:
    """Make sure downsampled_size is available for z,y,x and is not larger than data size.

    Parameters
    ----------
    data_size
        Size of data.
    downsampled_size
        Desired size of downsampled data

    Returns
    -------
        Size of downsampled data
    """
    if downsampled_size is None:
        return (data_size[-3], data_size[-2], data_size[-1])  # needed for mypy

    if isinstance(downsampled_size, int):
        downsampled_size = SpatialDimension(z=downsampled_size, y=downsampled_size, x=downsampled_size)

    return (
        min(downsampled_size.z, data_size[-3]),
        min(downsampled_size.y, data_size[-2]),
        min(downsampled_size.x, data_size[-1]),
    )


class CsmData(QData, init=False):
    """Coil sensitivity map class."""

    @classmethod
    def from_kdata_walsh(
        cls,
        kdata: KData,
        noise: KNoise | None = None,
        smoothing_width: int | SpatialDimension[int] = 5,
        chunk_size_otherdim: int | None = None,
        downsampled_size: int | SpatialDimension[int] | None = None,
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
        downsampled_size
            IData will be downsampled to this size before calculating the csm to speed up the calculation and
            reduce memory requirements. The final csm will be upsampled to the original size. If set to `None` no
            downsampling will be performed.

        Returns
        -------
            Coil sensitivity maps
        """
        from mrpro.algorithms.reconstruction import DirectReconstruction

        return cls.from_idata_walsh(
            DirectReconstruction(kdata, noise=noise, csm=None)(kdata),
            smoothing_width,
            chunk_size_otherdim,
            downsampled_size,
        )

    @classmethod
    def from_idata_walsh(
        cls,
        idata: IData,
        smoothing_width: int | SpatialDimension[int] = 5,
        chunk_size_otherdim: int | None = None,
        downsampled_size: int | SpatialDimension[int] | None = None,
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
        downsampled_size
            IData will be downsampled to this size before calculating the csm to speed up the calculation and
            reduce memory requirements. The final csm will be upsampled to the original size. If set to `None` no
            downsampling will be performed.


        Returns
        -------
            Coil sensitivity maps
        """
        from mrpro.algorithms.csm.walsh import walsh

        csm_fun = torch.vmap(
            lambda img: apply_lowres(
                lambda x: walsh(x, smoothing_width),
                size=get_downsampled_size(idata.data.shape, downsampled_size),
                dim=(-3, -2, -1),
            )(img),
            chunk_size=chunk_size_otherdim,
        )
        csm_tensor = csm_fun(idata.data.flatten(end_dim=-5)).reshape(idata.data.shape)
        # upsampled csm requires normalization
        csm_tensor = torch.nn.functional.normalize(csm_tensor, p=2, dim=-4, eps=1e-9)
        csm = cls(header=QHeader.from_iheader(idata.header), data=csm_tensor)
        return csm

    @classmethod
    def from_kdata_inati(
        cls,
        kdata: KData,
        noise: KNoise | None = None,
        smoothing_width: int | SpatialDimension[int] = 5,
        chunk_size_otherdim: int | None = None,
        downsampled_size: int | SpatialDimension[int] | None = None,
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
        downsampled_size
            IData will be downsampled to this size before calculating the csm to speed up the calculation and
            reduce memory requirements. The final csm will be upsampled to the original size. If set to `None` no
            downsampling will be performed.

        Returns
        -------
            Coil sensitivity maps
        """
        from mrpro.algorithms.reconstruction import DirectReconstruction

        return cls.from_idata_inati(
            DirectReconstruction(kdata, noise=noise, csm=None)(kdata),
            smoothing_width,
            chunk_size_otherdim,
            downsampled_size,
        )

    @classmethod
    def from_idata_inati(
        cls,
        idata: IData,
        smoothing_width: int | SpatialDimension[int] = 5,
        chunk_size_otherdim: int | None = None,
        downsampled_size: int | SpatialDimension[int] | None = None,
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
            Default is `None`, which means that all elements are processed at once.
        downsampled_size
            IData will be downsampled to this size before calculating the csm to speed up the calculation and
            reduce memory requirements. The final csm will be upsampled to the original size. If set to `None` no
            downsampling will be performed.

        Returns
        -------
            Coil sensitivity maps
        """
        from mrpro.algorithms.csm.inati import inati

        csm_fun = torch.vmap(
            lambda img: apply_lowres(
                lambda x: inati(x, smoothing_width),
                size=get_downsampled_size(idata.data.shape, downsampled_size),
                dim=(-3, -2, -1),
            )(img),
            chunk_size=chunk_size_otherdim,
        )
        csm_tensor = csm_fun(idata.data.flatten(end_dim=-5)).reshape(idata.data.shape)
        # upsampled csm requires normalization
        csm_tensor = torch.nn.functional.normalize(csm_tensor, p=2, dim=-4, eps=1e-9)
        csm = cls(header=QHeader.from_iheader(idata.header), data=csm_tensor)
        return csm

    def as_operator(self) -> SensitivityOp:
        """Create SensitivityOp using a copy of the CSMs."""
        from mrpro.operators.SensitivityOp import SensitivityOp

        return SensitivityOp(self)
