import torch
from torch.nn import Module

from mrpro.operators.ConjugateGradientOp import ConjugateGradientOp
from mrpro.operators.FourierOp import FourierOp


class ConjugateGradientDC(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, data: torch.Tensor, fourier_op: FourierOp):
        cg_op = ConjugateGradientOp(fourier_op)
