"""Coil Sensitivity Estimation."""

from mr2.algorithms.csm.walsh import walsh
from mr2.algorithms.csm.inati import inati
__all__ = ["inati", "walsh"]
