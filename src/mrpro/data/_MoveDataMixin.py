from __future__ import annotations

import dataclasses
from abc import ABC
from copy import deepcopy
from typing import Any
from typing import ClassVar
from typing import Protocol
from typing import Self
from typing import overload

import torch


class DataclassInstance(Protocol):
    """An instance of a dataclass."""

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]


class MoveDataMixin(ABC, DataclassInstance):
    """Move dataclass fields to cpu/gpu and convert dtypes."""

    data: torch.Tensor

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

        This will always return a new Data object.

        A torch.dtype and torch.device are inferred from the arguments
        of self.to(*args, **kwargs). Please have a look at the
        documentation of torch.Tensor.to() for more details.
        """
        kwargs_tensors = {**kwargs, 'copy': True}
        new_data: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            name = field.name
            data = getattr(self, name)
            if isinstance(data, torch.Tensor):
                new_data[name] = data.to(*args, **kwargs_tensors)
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
            device = torch.cuda.current_device()
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
    def device(self) -> torch.device:
        """Return the device of the data tensor."""
        return self.data.device
