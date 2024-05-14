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
        other_args: Sequence[Any] = ()
        other_kwargs: dict[str, Any] = {}
        dtype: torch.dtype | None = None
        device: torch.device | str | int | None = None

        # match dtype and device from args and kwargs
        match args, kwargs:
            case ((_dtype, *_args), {**_kwargs}) if isinstance(_dtype, torch.dtype):
                # overload 1
                dtype = _dtype
                other_args = _args
                other_kwargs = _kwargs
            case (_args, {'dtype': _dtype, **_kwargs}) if isinstance(_dtype, torch.dtype):
                # dtype as kwarg
                dtype = _dtype
                other_args = _args
                other_kwargs = _kwargs
            case ((other, *_args), {**_kwargs}) | (_args, {'other': other, **_kwargs}) if isinstance(
                other, torch.Tensor
            ):
                # overload 3: use dtype and device from other
                dtype = other.dtype
                device = other.device
        match args, kwargs:
            case ((_device, _dtype, *_args), {**_kwargs}) if isinstance(
                _device, torch.device | str | int | None
            ) and isinstance(_dtype, torch.dtype):
                # overload 2 with device and dtype
                dtype = _dtype
                device = _device
                other_args = _args
                other_kwargs = _kwargs
            case ((_device, *_args), {**_kwargs}) if isinstance(_device, torch.device | str | int | None):
                # overload 2, only device
                device = _device
                other_args = _args
                other_kwargs = _kwargs
            case (_args, {'device': _device, **_kwargs}) if isinstance(_device, torch.device | str | int | None):
                # device as kwarg
                device = _device
                other_args = _args
                other_kwargs = _kwargs

        other_kwargs['copy'] = True
        new_data: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            name = field.name
            data = getattr(self, name)
            if isinstance(data, torch.Tensor):
                new_device = data.device if device is None else device
                if dtype is None:
                    new_dtype = data.dtype
                elif data.dtype.is_floating_point:
                    new_dtype = dtype.to_real()
                elif data.dtype.is_complex:
                    new_dtype = dtype.to_complex()
                else:
                    # bool or int: keep as is
                    new_dtype = data.dtype
                new_data[name] = data.to(new_device, new_dtype, *other_args, **other_kwargs)
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
    ) -> Self:
        """Create copy of object with data in CUDA memory.

        This will always return a copy.


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
        if device is None:
            device = torch.device(torch.cuda.current_device())
        return self.to(device=device, memory_format=memory_format, non_blocking=non_blocking)

    def cpu(self, memory_format: torch.memory_format = torch.preserve_format) -> Self:
        """Create copy of object in CPU memory.

        This will always return a copy.


        Parameters
        ----------
        memory_format
            The desired memory format of returned tensor.
        """
        return self.to(device='cpu', memory_format=memory_format)

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
        return deepcopy(self)
