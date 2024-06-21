import torch

from mrpro.operators._LinearOperator import LinearOperator


class LambdaOp(LinearOperator):
    """Lambda Operator for regularization in iterative reconstruction.

    This operator multiplies the input tensor by a regularization scalar value (lambda).
    """

    def __init__(self, lambda_value: float):
        """Initialize the Lambda Operator.

        Parameters
        ----------
        lambda_value : float
            The regularization scalar value (lambda).
        """
        super().__init__()
        self.lambda_value = lambda_value

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Lambda Operator to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to which the operator is applied.

        Returns
        -------
        torch.Tensor
            The result of applying the Lambda Operator to the input tensor.
        """
        return self.lambda_value * x

    def H(self):
        """Return the adjoint of the Lambda Operator.

        Returns
        -------
        LambdaOp
            The adjoint of the Lambda Operator, which is itself since the operation is scalar multiplication.
        """
        return self

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        """Override the matmul operator to apply the Lambda Operator to another tensor.

        Parameters
        ----------
        other : torch.Tensor
            The tensor to which the Lambda Operator is applied.

        Returns
        -------
        torch.Tensor
            The result of applying the Lambda Operator to the input tensor.
        """
        return self._apply(other)
