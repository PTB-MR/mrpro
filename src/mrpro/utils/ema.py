"""Exponential Moving Average (EMA) dictionary."""

from collections.abc import ItemsView, KeysView, Mapping, ValuesView
from typing import Any

import torch


class EMADict:
    """
    Exponential Moving Average (EMA) dictionary.

    Maintains an EMA of values for each key. On update, existing keys are
    updated with EMA, and new keys are added directly.

    Detaches the values from the autograd graph.


    """

    def __init__(
        self,
        decay: float,
    ):
        """Initialize the EMA dictionary.

        Parameters
        ----------
        decay
            Decay rate for EMA (between 0 and 1).
        """
        self.decay: float = decay
        if not 0 <= decay <= 1:
            raise ValueError(f'Decay must be between 0 and 1, got {decay}')
        self._data: dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Get the value of the EMA dict for a given key."""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:  # noqa: ANN401
        """Set the value of the EMA dict for a given key."""
        if key in self._data:
            old_v = self._data[key]
            if isinstance(old_v, torch.Tensor) and isinstance(value, torch.Tensor):
                if torch.is_floating_point(old_v) or torch.is_complex(old_v):
                    old_v.mul_(self.decay).add_(value.detach().to(old_v.device), alpha=1.0 - self.decay)
                else:
                    old_v.copy_(value)
                return
            elif isinstance(old_v, float) and isinstance(value, float):  # noqa: SIM114
                self._data[key] = self.decay * old_v + (1.0 - self.decay) * value
                return
            elif isinstance(old_v, complex) and isinstance(value, complex):
                self._data[key] = self.decay * old_v + (1.0 - self.decay) * value
                return

        if isinstance(value, torch.Tensor):
            self._data[key] = value.detach().clone()
            return

        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete a key from the EMA dict."""
        del self._data[key]

    def __contains__(self, key: str) -> bool:
        """Check if a key is in the EMA dict."""
        return key in self._data

    def values(self) -> ValuesView[Any]:
        """Get the values of the EMA dict."""
        return self._data.values()

    def keys(self) -> KeysView[str]:
        """Get the keys of the EMA dict."""
        return self._data.keys()

    def items(self) -> ItemsView[str, Any]:
        """Get the items of the EMA dict."""
        return self._data.items()

    def update(self, other: Mapping[Any, Any]) -> None:
        """Update the EMA dict with another dictionary.

        For existing keys, performs EMA update. For new keys, adds them directly.

        Parameters
        ----------
        other
            Dictionary to update from.
        """
        for k, v in other.items():
            self.__setitem__(k, v)
