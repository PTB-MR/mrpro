import dataclasses
from copy import deepcopy

import torch

from mrpro.data import Data
from mrpro.operators import Operator


class DataOperator(torch.nn.Module):
    """Apply operator to the .data of a Data object.

    Wraps an existing Operator returns a new Data object with the result
    of the operator applied to the .data and a copy of all other
    attributes such as the Header.
    """

    def __init__(self, operator: Operator[torch.Tensor, tuple[torch.Tensor,]], return_type: type[Data] | None = None):
        """Initialize the DataOperator.

        Parameters
        ----------
        operator
            The operator to wrap
        return_type
            The type of the Data object to return. If None, the same type as the input is returned.
            Each field of the return type will we set to the value of the field with the same name in the input.
        """
        super().__init__()
        self.operator = operator
        self.return_type = return_type

    def forward(self, data: Data):
        """Apply the operator."""
        new_data = self.operator(data.data)

        properties = {
            field.name: deepcopy(getattr(data, field.name))
            for field in dataclasses.fields(data)
            if field.name != 'data'
        }

        if self.return_type is None:
            return type(data)(**properties, data=new_data)

        # match existing fields in properties to fields of the new class
        new_fields = [field.name for field in dataclasses.fields(self.return_type)]
        properties = {k: v for k, v in properties.items() if k in new_fields}
        return self.return_type(**properties, data=new_data)
