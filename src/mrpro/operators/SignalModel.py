"""Signal Model Operators."""

import torch
from typing_extensions import TypeVarTuple, Unpack

from mrpro.operators.Operator import Operator

Tin = TypeVarTuple('Tin')


# SignalModel has multiple inputs and one output
class SignalModel(Operator[Unpack[Tin], tuple[torch.Tensor,]]):
    """Signal Model Operator."""
