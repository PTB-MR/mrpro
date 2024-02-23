import dataclasses
from copy import deepcopy

import torch

from mrpro.data import Data
from mrpro.operators import Operator


class DataOperator(torch.nn.Module):
    """Apply operator to the .data of a Data object.

    returns a new Data object with the result of the operator applied to
    the .data

    # TODO: THIS IS ONLY FOR TESTING, REMOVE LATER

    DO NOT MERGE INTO MAIN

    #
    """

    def __init__(self, Operator: Operator[torch.Tensor, tuple[torch.Tensor,]], returntype: type[Data] | None = None):
        """Initialize the DataOperator.

        Parameters
        ----------
        Operator
            The operator to apply
        returntype
            The type of the Data object to return. If None, the same type as the input is returned.
        """
        super().__init__()
        self.Operator = Operator
        self.returntype = returntype

    def forward(self, data: Data):
        """Apply the operator."""
        ret = self.Operator(data.data)
        pref = {
            field.name: deepcopy(getattr(data, field.name))
            for field in dataclasses.fields(data)
            if field.name != 'data'
        }
        if self.returntype is None:
            return type(data)(**pref, data=ret)
        else:
            newfields = [field.name for field in dataclasses.fields(self.returntype)]
            pref = {k: v for k, v in pref.items() if k in newfields}
            return self.returntype(**pref, data=ret)
