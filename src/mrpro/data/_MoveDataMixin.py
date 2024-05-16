"""MoveDataMixin."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
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
from collections.abc import Sequence
from copy import deepcopy
from typing import Any
from typing import ClassVar
from typing import Protocol
from typing import Self
from typing import overload

import torch


class InconsistentDeviceError(ValueError):
    def __init__(self, *devices):
        super().__init__(f'Inconsistent devices found, found at least {", ".join(str(d) for d in devices)}')


class DataclassInstance(Protocol):
    """An instance of a dataclass."""

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]


class MoveDataMixin(ABC, DataclassInstance):
    """Move dataclass fields to cpu/gpu and convert dtypes."""

    @overload
    def to(
        self, dtype: torch.dtype, non_blocking: bool = False, *, memory_format: torch.memory_format | None = None
    ) -> Self: ...

    @overload
    def to(
        self,
        device: str | torch.device | int | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        *,
        memory_format: torch.memory_format | None = None,
    ) -> Self: ...

    @overload
    def to(
        self, other: torch.Tensor, non_blocking: bool = False, *, memory_format: torch.memory_format | None = None
    ) -> Self: ...

    def to(self, *args, **kwargs) -> Self:
        """Perform dtype and/or device conversion of data.

        This will always return a new Data object with
        all tensors copied, even if no conversion is necessary.

        A torch.dtype and torch.device are inferred from the arguments
        of self.to(*args, **kwargs). Please have a look at the
        documentation of torch.Tensor.to() for more details.

        The conversion will be applied to all Tensor fields of the dataclass,
        and to all fields that implement the MoveDataMixin.

        The dtype-type, i.e. float/complex will always be preserved,
        but the precision of floating point dtypes might be changed.

        Example:
            If called with dtype=torch.float32 OR dtype=torch.complex64:
                - A complex128 tensor will be converted to complex64
                - A float64 tensor will be converted to float32
                - A bool tensor will remain bool
                - An int64 tensor will remain int64

        If other conversions are desired, please use the torch.Tensor.to() method of
        the fields directly.
        """
        def parse3(other, non_blocking=False, copy=False):
            return other.device, other.dtype, non_blocking, copy, torch.preserve_format
        def parse1(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format):
            return None, dtype, non_blocking, copy, memory_format
        def parse2(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format):
            return device, dtype, non_blocking, copy, memory_format
        if args and isinstance(args[0],torch.Tensor) or 'other' in kwargs:
            parser = parse3
        elif args and isinstance(args[0], torch.dtype):
            parser= parse1
        else:
            parser = parse2
        device, dtype, non_blocking, copy, memory_format = parser(*args,**kwargs)
        return self._to(device=device, dtype=dtype, non_blocking=non_blocking, memory_format=memory_format, copy=copy, memo={})


    def _to(device: torch.device | str | int | None, dtype: torch.dtype | None, non_blocking: bool, memory_format: torch.memory_format | None, copy:bool, memo:dict) -> Self:
        new_data: dict[str, Any] = {}

        def _tensor_to(data: torch.Tensor) -> torch.Tensor:
            """Move tensor to device and convert dtype if necessary."""
            if dtype is not None and data.dtype.is_floating_point:
                    new_dtype = dtype.to_real()
            if dtype is not None and data.dtype.is_complex:
                    new_dtype = dtype.to_complex()
            else:
                    # bool or int: keep as is
                    new_dtype = None
            return data.to(device, new_dtype, *other_args, **{**other_kwargs, 'copy':True})

        for field in dataclasses.fields(self):
            name = field.name
            data = getattr(self, name)
            if isinstance(data, torch.Tensor):
                new_data[name] = _tensor_to(data)
            elif isinstance(data, torch.nn.Module):
                data.apply(_tensor_to)



            elif isinstance(data, MoveDataMixin):
                new_data[name] = data.to(*args, **kwargs)
            else:
                new_data[name] = deepcopy(data)
        return type(self)(**new_data)

    def cuda(
        self,
        device: torch.device | str | int | None = None,
        non_blocking: bool = False,
        memory_format: torch.memory_format = torch.preserve_format,
        copy: bool = True,
    ) -> Self:
        """Put object in CUDA memory.


        Parameters
        ----------
        device
            The destination GPU device. Defaults to the current CUDA device.
        non_blocking
            If True and the source is in pinned memory, the copy will be asynchronous with respect to the host.
            Otherwise, the argument has no effect.
        memory_format
            The desired memory format of returned tensor.
        copy
            If True, the returned tensor will always be a copy, even if the input was already on the correct device.
        """
        if device is None:
            device = torch.device(torch.cuda.current_device())
        return self.to(device=device, memory_format=memory_format, non_blocking=non_blocking, copy=copy)

    def cpu(self, memory_format: torch.memory_format = torch.preserve_format, copy:bool=True) -> Self:
        """Put in CPU memory.


        Parameters
        ----------
        memory_format
            The desired memory format of returned tensor.
        copy
            If True, the returned tensor will always be a copy, even if the input was already on the correct device.
        """
        return self.to(device='cpu', memory_format=memory_format,copy=copy)

    @property
    def device(self) -> torch.device | None:
        """Return the device of the tensors.

        Looks at each field of a dataclass implementing a device attribute,
        such as torch.Tensors or MoveDataMixin instances. If the devices
        of the fields differ, an InconsistentDeviceError is raised, otherwise
        the device is returned. If no field implements a device attribute,
        None is returned.

        Raises
        ------
        InconsistentDeviceError:
            If the devices of different fields differ.

        Returns
        -------
            The device of the fields or None if no field implements a device attribute.
        """
        device: None | torch.device = None
        for field in dataclasses.fields(self):
            data = getattr(self, field.name)
            if not hasattr(data, 'device'):
                continue
            current_device = getattr(data, 'device', None)
            if current_device is None:
                continue
            if device is None:
                device = current_device
            elif device != current_device:
                raise InconsistentDeviceError(current_device, device)
        return device

    def clone(self: Self) -> Self:
        """Return a deep copy of the object."""
        return self._to(device=None, dtype=None, non_blocking=False, memory_format=torch.preserve_format, copy=True, memo={})

    @property
    def is_cuda(self) -> bool:
        """Return True if all tensors are on a single CUDA device.

        Checks all tensor attributes of the dataclass for their device,
        (recursively if an attribute is a MoveDataMixin)


        Returns False if not all tensors are on the same CUDA devices, or if the device is inconsistent,
        returns True if the data class has no tensors as attributes.
        """
        try:
            device = self.device
        except InconsistentDeviceError:
            return False
        if device is None:
            return True
        return device.type == 'cuda'

    @property
    def is_cpu(self) -> bool:
        """Return True if all tensors are on the CPU.

        Checks all tensor attributes of the dataclass for their device,
        (recursively if an attribute is a MoveDataMixin)

        Returns False if not all tensors are on cpu or if the device is inconsistent,
        returns True if the data class has no tensors as attributes.
        """
        try:
            device = self.device
        except InconsistentDeviceError:
            return False
        if device is None:
            return True
        return device.type == 'cpu'
