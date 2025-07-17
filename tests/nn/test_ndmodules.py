"""Tests for the ndmodules module."""

import pytest
import torch
from mrpro.nn.ndmodules import (
    AdaptiveAvgPoolND,
    AvgPoolND,
    BatchNormND,
    ConvND,
    ConvTransposeND,
    InstanceNormND,
    MaxPoolND,
)


def test_convnd() -> None:
    """Test ConvND."""
    assert ConvND(1) is torch.nn.Conv1d
    assert ConvND(2) is torch.nn.Conv2d
    assert ConvND(3) is torch.nn.Conv3d
    with pytest.raises(NotImplementedError):
        ConvND(4)


def test_convtransposend() -> None:
    """Test ConvTransposeND."""
    assert ConvTransposeND(1) is torch.nn.ConvTranspose1d
    assert ConvTransposeND(2) is torch.nn.ConvTranspose2d
    assert ConvTransposeND(3) is torch.nn.ConvTranspose3d
    with pytest.raises(NotImplementedError):
        ConvTransposeND(4)


def test_maxpoolnd() -> None:
    """Test MaxPoolND."""
    assert MaxPoolND(1) is torch.nn.MaxPool1d
    assert MaxPoolND(2) is torch.nn.MaxPool2d
    assert MaxPoolND(3) is torch.nn.MaxPool3d
    with pytest.raises(NotImplementedError):
        MaxPoolND(4)


def test_avgpoolnd() -> None:
    """Test AvgPoolND."""
    assert AvgPoolND(1) is torch.nn.AvgPool1d
    assert AvgPoolND(2) is torch.nn.AvgPool2d
    assert AvgPoolND(3) is torch.nn.AvgPool3d
    with pytest.raises(NotImplementedError):
        AvgPoolND(4)


def test_adaptiveavgpoolnd() -> None:
    """Test AdaptiveAvgPoolND."""
    assert AdaptiveAvgPoolND(1) is torch.nn.AdaptiveAvgPool1d
    assert AdaptiveAvgPoolND(2) is torch.nn.AdaptiveAvgPool2d
    assert AdaptiveAvgPoolND(3) is torch.nn.AdaptiveAvgPool3d
    with pytest.raises(NotImplementedError):
        AdaptiveAvgPoolND(4)


def test_instancenormnd() -> None:
    """Test InstanceNormND."""
    assert InstanceNormND(1) is torch.nn.InstanceNorm1d
    assert InstanceNormND(2) is torch.nn.InstanceNorm2d
    assert InstanceNormND(3) is torch.nn.InstanceNorm3d
    with pytest.raises(NotImplementedError):
        InstanceNormND(4)


def test_batchnormnd() -> None:
    """Test BatchNormND."""
    assert BatchNormND(1) is torch.nn.BatchNorm1d
    assert BatchNormND(2) is torch.nn.BatchNorm2d
    assert BatchNormND(3) is torch.nn.BatchNorm3d
    with pytest.raises(NotImplementedError):
        BatchNormND(4)
