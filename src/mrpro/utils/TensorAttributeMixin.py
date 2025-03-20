"""Mixin for smarter tensor attributes."""

from typing import Any

import torch


class TensorAttributeMixin(torch.nn.Module):
    """Create tensor attributes as buffer."""

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        """Set attribute.

        Set tensors not requiring gradients as buffer.

        Parameters
        ----------
        name
            name of the attribute.
        value
            attribute to set.
        """
        if isinstance(value, torch.Tensor) and not isinstance(value, torch.nn.Parameter) and not value.requires_grad:
            self.register_buffer(name, value)
        else:
            super().__setattr__(name, value)
