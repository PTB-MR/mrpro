"""Base class for data objects."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import dataclasses
from abc import ABC
from copy import deepcopy
from typing import Any
from typing import Self

import torch

from mrpro.data.MoveDataMixin import MoveDataMixin


@dataclasses.dataclass(slots=True, frozen=True)
class Data(MoveDataMixin, ABC):
    """A general data class with field data and header."""

    data: torch.Tensor
    header: Any
<<<<<<< HEAD
=======

    def to(self, *args, **kwargs) -> Self:
        """Perform dtype and/or device conversion of data.

        A torch.dtype and torch.device are inferred from the arguments
        of self.to(*args, **kwargs). Please have a look at the
        documentation of torch.Tensor.to() for more details.
        """
        header = deepcopy(self.header)
        data = self.data.to(*args, **kwargs)
        return type(self)(header=header, data=data)

    def cuda(
        self,
        device: torch.device | None = None,
        non_blocking: bool = False,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> Self:
        """Create copy of object with data in CUDA memory.

        Parameters
        ----------
        device
            The destination GPU device. Defaults to the current CUDA device.
        non_blocking
            If True and the source is in pinned memory, the copy will be asynchronous with respect to the host.
            Otherwise, the argument has no effect.
        memory_format
            The desired memory format of returned tensor.
        """
        header = deepcopy(self.header)
        data = self.data.cuda(device=device, non_blocking=non_blocking, memory_format=memory_format)
        return type(self)(header=header, data=data)

    def cpu(self, memory_format: torch.memory_format = torch.preserve_format) -> Self:
        """Create copy of object in CPU memory.

        Parameters
        ----------
        memory_format
            The desired memory format of returned tensor.
        """
        header = deepcopy(self.header)
        data = self.data.cpu(memory_format=memory_format)
        return type(self)(header=header, data=data)
>>>>>>> 39605f1 (some type fixes maybe)
