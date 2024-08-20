"""Signal Model Operators."""

from typing import TypeVarTuple

import torch

from mrpro.operators.Operator import Operator

Tin = TypeVarTuple('Tin')


# SignalModel has multiple inputs and one output
class SignalModel(Operator[*Tin, tuple[torch.Tensor,]]):
    """Signal Model Operator."""

    @staticmethod
    def expand_tensor_dim(parameter: torch.Tensor, n_dim_to_expand: int) -> torch.Tensor:
        """Extend the number of dimensions of a parameter tensor.

        This is commonly used in the `model.forward` to ensure the model parameters can be broadcasted to the
        quantitative maps. E.g. a simple `InversionRecovery` model is evaluated for six different inversion times `ti`.
        The inversion times are commonly the same for each voxel and hence `ti` could be of shape (6,) and the T1 and M0
        map could be of shape (100,100,100). To make sure `ti` can be broadcasted to the maps it needs to be extended to
        the shape (6,1,1,1) which then yields a signal of shape (6,100,100,100).

        Parameters
        ----------
        parameter
            Parameter (e.g with shape (m,n))
        n_dim_to_expand
            Number of dimensions to expand. If <= 0 then parameter is not changed.

        Returns
        -------
            Parameter with expanded dimensions (e.g. (m,n,1,1) for n_dim_to_expand = 2)
        """
        return parameter[..., *[None] * (n_dim_to_expand)] if n_dim_to_expand > 0 else parameter
