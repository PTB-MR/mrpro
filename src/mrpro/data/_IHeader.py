"""MR image data header (IHeader) dataclass."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import annotations

from dataclasses import dataclass

from mrpro.data._KHeader import KHeader
from mrpro.data._SpatialDimension import SpatialDimension


@dataclass(slots=True)
class IHeader:
    """MR image data header.

    All information that is not covered by the dataclass is stored in
    the misc dict. Our code shall not rely on this information, and it
    is not guaranteed to be present. Also, the information in the misc
    dict is not guaranteed to be correct or tested.
    """

    # ToDo: decide which attributes to store in the header
    # ToDo: decide on types. Allow None ?
    fov: SpatialDimension[float] | None
    te: list[float] | None
    ti: list[float] | None
    fa: list[float] | None
    tr: list[float] | None

    @classmethod
    def from_kheader(cls, kheader: KHeader):
        """Create IHeader object from KHeader object.

        Parameters
        ----------
        kheader
            MR raw data header (KHeader) containing required meta data.
        """

        return cls(
            fov=kheader.encoding_fov,  # rename to fov
            te=kheader.te,
            ti=kheader.ti,
            fa=kheader.fa,
            tr=kheader.tr,
        )
