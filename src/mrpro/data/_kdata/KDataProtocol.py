"""Protocol for KData."""

from typing import Literal

import torch
from typing_extensions import Protocol, Self

from mrpro.data.KHeader import KHeader
from mrpro.data.KTrajectory import KTrajectory


class _KDataProtocol(Protocol):
    """Protocol for KData used for type hinting in KData mixins.

    Note that the actual KData class can have more properties and methods than those defined here.

    If you want to use a property or method of KData in a new KDataMixin class,
    you must add it to this Protocol to make sure that the type hinting works [PRO]_.

    References
    ----------
    .. [PRO] Protocols https://typing.readthedocs.io/en/latest/spec/protocol.html#protocols
    """

    @property
    def header(self) -> KHeader: ...

    @property
    def data(self) -> torch.Tensor: ...

    @property
    def traj(self) -> KTrajectory: ...

    def __init__(self, header: KHeader, data: torch.Tensor, traj: KTrajectory): ...

    def _split_k2_or_k1_into_other(
        self,
        split_idx: torch.Tensor,
        other_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
        split_dir: Literal['k1', 'k2'],
    ) -> Self: ...
