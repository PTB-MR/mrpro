"""Class for coil sensitivity maps (csm)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from mr2.data.IData import IData
from mr2.data.QData import QData
from mr2.data.QHeader import QHeader
from mr2.data.SpatialDimension import SpatialDimension
from mr2.utils.interpolate import apply_lowres
from mr2.utils.smap import smap

if TYPE_CHECKING:
    from mr2.data.KData import KData
    from mr2.data.KNoise import KNoise
    from mr2.operators.SensitivityOp import SensitivityOp


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
    def from_kdata_espirit(
        cls,
        acs: KData,
        singular_value_threshold: float = 0.02,
        kernel_width: int = 6,
        crop_threshold: float = 0.3,
    ) -> CsmData:
        """Espirit sensitivity Estimation.

        Estimate the coil sensitivity maps from the auto-calibration data of a
        cartesian acquisitions.

        Parameters
        ----------
        acs
            fully sampled auto-calibration data in the center of k-space
        singular_value_threshold
            threshold for the singular value decomposition
        kernel_width
            width of the kernel for the espirit algorithm
        crop_threshold
            threshold for the crop of the espirit algorithm

        """
        from mr2.algorithms.csm.espirit import espirit

        # TODO: check that the data is fully sampled and cartesian

        csm_data = smap(
            lambda c: espirit(
                c,
                img_shape=acs.header.recon_matrix,
                singular_value_threshold=singular_value_threshold,
                kernel_width=kernel_width,
                crop_threshold=crop_threshold,
                n_iterations=10,
            ),
            acs.data,
            passed_dimensions=(-4, -3, -2, -1),  # coils, z, y, x
        )
        return cls(header=acs.header, data=csm_data)

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

        See also `~mr2.algorithms.csm.walsh`.

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
        from mr2.algorithms.reconstruction import DirectReconstruction

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

        See also `~mr2.algorithms.csm.walsh`.

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
        from mr2.algorithms.csm.walsh import walsh

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

        See also `~mr2.algorithms.csm.inati`.

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
        from mr2.algorithms.reconstruction import DirectReconstruction

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

        See also `~mr2.algorithms.csm.inati`.

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
        from mr2.algorithms.csm.inati import inati

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
        from mr2.operators.SensitivityOp import SensitivityOp

        return SensitivityOp(self)
