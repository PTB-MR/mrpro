"""SpatialDimension dataclass."""

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
from typing import Callable
from typing import Generic
from typing import Protocol
from typing import TypeVar

import numpy as np
import torch
from numpy.typing import ArrayLike

T = TypeVar('T', int, float, torch.Tensor)


class XYZ(Protocol[T]):
    x: T
    y: T
    z: T


@dataclass(slots=True)
class SpatialDimension(Generic[T]):
    """Spatial dataclass."""

    x: T
    y: T
    z: T

    @classmethod
    def from_xyz(cls, data: XYZ[T], conversion: Callable[[T], T] | None = None) -> SpatialDimension[T]:
        """Create a SpatialDimension from something with (.x .y .z) parameters
        that are either float, int or Any.

        Parameters:
        --------
        data: should implement .x .y .z. For example ismrmrd's matrixSizeType.
        conversion (optional): will be called for each value to convert it
        """
        if conversion is not None:
            return cls(conversion(data.x), conversion(data.y), conversion(data.z))
        return cls(data.x, data.y, data.z)

    @classmethod
    def from_array(
        cls, data: ArrayLike, conversion: Callable[[torch.Tensor], torch.Tensor] | None = None
    ) -> SpatialDimension[torch.Tensor]:
        """Create a SpatialDimension from something with an arraylike
        interface.

        Parameters:
        --------
        data: shape (..., 3)
        conversion (optional): will be called for each value to convert it
        """
        if not isinstance(data, (np.ndarray, torch.Tensor)):
            data = np.asarray(data)
        data = np.asarray(data)
        if np.size(data, -1) != 3:
            raise ValueError(f'Expected last dimension to be 3, got {np.size(data, -1)}')

        x = torch.as_tensor(data[..., 0])
        y = torch.as_tensor(data[..., 1])
        z = torch.as_tensor(data[..., 2])

        if conversion is not None:
            x = conversion(x)
            y = conversion(y)
            z = conversion(z)
        return cls(x, y, z)
