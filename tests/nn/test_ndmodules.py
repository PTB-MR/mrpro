"""Tests for the ndmodules module."""

import pytest
import torch
from mr2.nn.ndmodules import (
    adaptiveAvgPoolND,
    avgPoolND,
    batchNormND,
    convND,
    convTransposeND,
    instanceNormND,
    maxPoolND,
)


def test_convnd() -> None:
    """Test ConvND."""
    assert convND(1) is torch.nn.Conv1d
    assert convND(2) is torch.nn.Conv2d
    assert convND(3) is torch.nn.Conv3d
    with pytest.raises(NotImplementedError):
        convND(4)


def test_convtransposend() -> None:
    """Test ConvTransposeND."""
    assert convTransposeND(1) is torch.nn.ConvTranspose1d
    assert convTransposeND(2) is torch.nn.ConvTranspose2d
    assert convTransposeND(3) is torch.nn.ConvTranspose3d
    with pytest.raises(NotImplementedError):
        convTransposeND(4)


def test_maxpoolnd() -> None:
    """Test MaxPoolND."""
    assert maxPoolND(1) is torch.nn.MaxPool1d
    assert maxPoolND(2) is torch.nn.MaxPool2d
    assert maxPoolND(3) is torch.nn.MaxPool3d
    with pytest.raises(NotImplementedError):
        maxPoolND(4)


def test_avgpoolnd() -> None:
    """Test AvgPoolND."""
    assert avgPoolND(1) is torch.nn.AvgPool1d
    assert avgPoolND(2) is torch.nn.AvgPool2d
    assert avgPoolND(3) is torch.nn.AvgPool3d
    with pytest.raises(NotImplementedError):
        avgPoolND(4)


def test_adaptiveavgpoolnd() -> None:
    """Test AdaptiveAvgPoolND."""
    assert adaptiveAvgPoolND(1) is torch.nn.AdaptiveAvgPool1d
    assert adaptiveAvgPoolND(2) is torch.nn.AdaptiveAvgPool2d
    assert adaptiveAvgPoolND(3) is torch.nn.AdaptiveAvgPool3d
    with pytest.raises(NotImplementedError):
        adaptiveAvgPoolND(4)


def test_instancenormnd() -> None:
    """Test InstanceNormND."""
    assert instanceNormND(1) is torch.nn.InstanceNorm1d
    assert instanceNormND(2) is torch.nn.InstanceNorm2d
    assert instanceNormND(3) is torch.nn.InstanceNorm3d
    with pytest.raises(NotImplementedError):
        instanceNormND(4)


def test_batchnormnd() -> None:
    """Test BatchNormND."""
    assert batchNormND(1) is torch.nn.BatchNorm1d
    assert batchNormND(2) is torch.nn.BatchNorm2d
    assert batchNormND(3) is torch.nn.BatchNorm3d
    with pytest.raises(NotImplementedError):
        batchNormND(4)
