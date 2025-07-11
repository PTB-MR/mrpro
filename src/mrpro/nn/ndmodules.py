"""Helper functions to get the correct N-dimensional module."""

import torch


def ConvND(dim: int) -> type[torch.nn.Conv1d] | type[torch.nn.Conv2d] | type[torch.nn.Conv3d]:  # noqa: N802
    """Get the `dim`-dimensional convolution class.

    Parameters
    ----------
    dim
        The dimension of the convolution.

    Returns
    -------
        The convolution class.
    """
    match dim:
        case 1:
            return torch.nn.Conv1d
        case 2:
            return torch.nn.Conv2d
        case 3:
            return torch.nn.Conv3d
        case _:
            raise NotImplementedError(f'ConvND for dim {dim} not implemented. Raise an issue if you need this.')


def ConvTransposeND(  # noqa: N802
    dim: int,
) -> type[torch.nn.ConvTranspose1d] | type[torch.nn.ConvTranspose2d] | type[torch.nn.ConvTranspose3d]:
    """Get the `dim`-dimensional transposed convolution class.

    Parameters
    ----------
    dim
        The dimension of the transposed convolution.

    Returns
    -------
        The transposed convolution class.
    """
    match dim:
        case 1:
            return torch.nn.ConvTranspose1d
        case 2:
            return torch.nn.ConvTranspose2d
        case 3:
            return torch.nn.ConvTranspose3d
        case _:
            raise NotImplementedError(
                f'ConvTransposeND for dim {dim} not implemented. Raise an issue if you need this.'
            )


def MaxPoolND(dim: int) -> type[torch.nn.MaxPool1d] | type[torch.nn.MaxPool2d] | type[torch.nn.MaxPool3d]:  # noqa: N802
    """Get the `dim`-dimensional max pooling class.

    Parameters
    ----------
    dim
        The dimension of the max pooling.

    Returns
    -------
        The max pooling class.
    """
    match dim:
        case 1:
            return torch.nn.MaxPool1d
        case 2:
            return torch.nn.MaxPool2d
        case 3:
            return torch.nn.MaxPool3d
        case _:
            raise NotImplementedError(f'MaxPoolNd for dim {dim} not implemented. Raise an issue if you need this.')


def AvgPoolND(dim: int) -> type[torch.nn.AvgPool1d] | type[torch.nn.AvgPool2d] | type[torch.nn.AvgPool3d]:  # noqa: N802
    """Get the `dim`-dimensional average pooling class.

    Parameters
    ----------
    dim
        The dimension of the average pooling.

    Returns
    -------
        The average pooling class.
    """
    match dim:
        case 1:
            return torch.nn.AvgPool1d
        case 2:
            return torch.nn.AvgPool2d
        case 3:
            return torch.nn.AvgPool3d
        case _:
            raise NotImplementedError(f'AvgPoolNd for dim {dim} not implemented. Raise an issue if you need this.')


def AdaptiveAvgPoolND(  # noqa: N802
    dim: int,
) -> type[torch.nn.AdaptiveAvgPool1d] | type[torch.nn.AdaptiveAvgPool2d] | type[torch.nn.AdaptiveAvgPool3d]:
    """Get the `dim`-dimensional adaptive average pooling class.

    Parameters
    ----------
    dim
        The dimension of the adaptive average pooling.

    Returns
    -------
        The adaptive average pooling class.
    """
    match dim:
        case 1:
            return torch.nn.AdaptiveAvgPool1d
        case 2:
            return torch.nn.AdaptiveAvgPool2d
        case 3:
            return torch.nn.AdaptiveAvgPool3d
        case _:
            raise NotImplementedError(
                f'AdaptiveAvgPoolNd for dim {dim} not implemented. Raise an issue if you need this.'
            )


def InstanceNormND(  # noqa: N802
    dim: int,
) -> type[torch.nn.InstanceNorm1d] | type[torch.nn.InstanceNorm2d] | type[torch.nn.InstanceNorm3d]:
    """Get the `dim`-dimensional instance normalization class.

    Parameters
    ----------
    dim
        The dimension of the instance normalization.

    Returns
    -------
        The instance normalization class.
    """
    match dim:
        case 1:
            return torch.nn.InstanceNorm1d
        case 2:
            return torch.nn.InstanceNorm2d
        case 3:
            return torch.nn.InstanceNorm3d
        case _:
            raise NotImplementedError(f'InstanceNormNd for dim {dim} not implemented. Raise an issue if you need this.')


def BatchNormND(  # noqa: N802
    dim: int,
) -> type[torch.nn.BatchNorm1d] | type[torch.nn.BatchNorm2d] | type[torch.nn.BatchNorm3d]:
    """Get the `dim`-dimensional batch normalization class.

    Parameters
    ----------
    dim
        The dimension of the batch normalization.

    Returns
    -------
        The batch normalization class.
    """
    match dim:
        case 1:
            return torch.nn.BatchNorm1d
        case 2:
            return torch.nn.BatchNorm2d
        case 3:
            return torch.nn.BatchNorm3d
        case _:
            raise NotImplementedError(f'BatchNormNd for dim {dim} not implemented. Raise an issue if you need this.')
