"""Signal Model Operators."""

from typing import TypeVarTuple

import torch

from mrpro.operators.Operator import Operator

Tin = TypeVarTuple('Tin')


# SignalModel has multiple inputs and one output
class SignalModel(Operator[*Tin, tuple[torch.Tensor,]]):
    """Signal Model Operator."""
