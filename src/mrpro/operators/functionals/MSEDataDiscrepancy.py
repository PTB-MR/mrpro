"""Mean squared error (MSE) data-discrepancy function."""

import torch
import torch.nn.functional as F  # noqa: N812

from mrpro.operators.Operator import Operator


class MSEDataDiscrepancy(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """Mean Squared Error (MSE) loss function.

    This class implements the function :math:`1./N * || . - data ||_2^2`, where :math:`N` equals to the number of
    elements of the tensor.

    Note: if one of data or input is complex-valued, we identify the space :math:`C^N` with :math:`R^{2N}` and
    multiply the output by 2. By this, we achieve that for example :math:`MSE(1)` = :math:`MSE(1+1j*0)` = 1.

    Parameters
    ----------
    data
        observed data
    """

    def __init__(self, data: torch.Tensor):
        """Initialize the MSE data-discrepancy operator."""
        super().__init__()

        # observed data
        self.data = data

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Calculate the MSE of the input.

        Parameters
        ----------
        x
            tensor whose MSE with respect to the data given at initialization should be calculated

        Returns
        -------
            Mean Squared Error (MSE) of input and the data
        """
        if torch.is_complex(x) or torch.is_complex(self.data):
            # F.mse_loss is only implemented for real tensors
            # Thus, we cast both to C and then to R^2
            # and undo the division by ten twice the number of elements in mse_loss
            x_r2 = torch.view_as_real(x) if torch.is_complex(x) else torch.view_as_real(x + 1j * 0)
            data_r2 = (
                torch.view_as_real(self.data) if torch.is_complex(self.data) else torch.view_as_real(self.data + 1j * 0)
            )
            mse = F.mse_loss(x_r2, data_r2) * 2.0
        else:  # both are real
            mse = F.mse_loss(x, self.data)
        return (mse,)
