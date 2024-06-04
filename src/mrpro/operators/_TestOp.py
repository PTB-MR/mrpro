import torch


class ShowCaseClass:
    def __init__(self, a: int):
        self.a = a

    def correct_method(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Return tuple of input tensor."""
        return (x,)

    def wrong_docstring(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Return tuple of input tensor

        As expected, ruff complains about D400 (missing period 1st line).
        """
        return (x,)

    def no_type_annotations(self, x):
        """Return tuple of input tensor.

        As expected, ruff complains about ANN001 (missing type ann).
        """
        return (x,)

    # ruff does not detect missing docstring
    # ruff does not detect missing return type annotation
    def no_return_type_no_docstring(self, x: torch.Tensor):
        return (x,)


def another_function(x):  # ruff only detects missing type annotation
    return (x,)
