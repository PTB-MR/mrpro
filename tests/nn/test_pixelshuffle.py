"""Test PixelShuffle and PixelUnshuffle."""

import torch
from mrpro.nn.PixelShuffle import PixelShuffle, PixelShuffleUpsample, PixelUnshuffle, PixelUnshuffleDownsample
from mrpro.utils import RandomGenerator


def test_pixel_shuffle_2d():
    """Test PixelUnshuffle's fast path for 2D images."""
    x = torch.arange(3 * 4 * 8).reshape(1, 3, 4, 8)
    pixel_unshuffle = PixelUnshuffle(2)
    y = pixel_unshuffle(x)
    assert y.shape == (1, 3 * 4, 4 // 2, 8 // 2)

    pixel_shuffle = PixelShuffle(2)
    z = pixel_shuffle(y)
    assert z.shape == (1, 3, 4, 8)
    assert (x == z).all()


def test_pixel_unshuffle_4d():
    """Test PixelUnshuffle's general case."""
    x = torch.arange(3 * 4 * 8 * 10 * 12).reshape(1, 3, 4, 8, 10, 12)
    pixel_unshuffle = PixelUnshuffle(2)
    y = pixel_unshuffle(x)
    assert y.shape == (1, 3 * 16, 4 // 2, 8 // 2, 10 // 2, 12 // 2)

    pixel_shuffle = PixelShuffle(2)
    z = pixel_shuffle(y)
    assert z.shape == (1, 3, 4, 8, 10, 12)
    assert (x == z).all()


def test_pixelunshuffle_features_last():
    """Test PixelUnshuffle with features_last."""
    x = torch.arange(3 * 4 * 8 * 10 * 12).reshape(1, 3, 4, 8, 10, 12)
    pixel_unshuffle_last = PixelUnshuffle(2, features_last=True)
    pixel_unshuffle = PixelUnshuffle(2, features_last=False)
    y_last = pixel_unshuffle_last(x.moveaxis(1, -1)).moveaxis(-1, 1)
    y_normal = pixel_unshuffle(x)
    assert (y_last == y_normal).all()


def test_pixelshuffle_features_last():
    """Test PixelS	huffle with features_last."""
    x = torch.arange(3 * 4 * 8 * 10 * 12).reshape(1, -1, 2, 4, 5, 6)
    pixel_shuffle_last = PixelShuffle(2, features_last=True)
    pixel_shuffle = PixelShuffle(2, features_last=False)
    y_last = pixel_shuffle_last(x.moveaxis(1, -1)).moveaxis(-1, 1)
    y_normal = pixel_shuffle(x)
    assert (y_last == y_normal).all()


def test_unpixelshuffledownsample_residual():
    """Test PixelUnshuffleDownsample with residual."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 2, 9, 12, 15))
    downsample = PixelUnshuffleDownsample(3, 2, 27, downscale_factor=3, residual=True)
    y = downsample(x)
    assert y.shape == (1, 27, 3, 4, 5)


def test_pixelshuffleupsample_residual():
    """Test PixelShuffleUpsample with residual."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 2, 3, 4, 5))
    upsample = PixelShuffleUpsample(3, 2, 1, upscale_factor=3, residual=True)
    y = upsample(x)
    assert y.shape == (1, 1, 9, 12, 15)


def test_pixelshuffleupsample_pixelunshuffledownsample():
    """Test if PixelUnshuffleDownsample is the inverse of PixelShuffleUpsample."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 3**3, 3, 4, 5))
    # Only without residual, the upsample and downsample are inverses.
    downsample = PixelUnshuffleDownsample(3, 1, 3**3, downscale_factor=3, residual=False)
    upsample = PixelShuffleUpsample(3, 3**3, 1, upscale_factor=3, residual=False)
    # Only if the convs are Identity, the upsample and downsample are inverses.
    torch.nn.init.dirac_(downsample.conv.weight)
    torch.nn.init.dirac_(upsample.conv.weight)
    torch.nn.init.zeros_(downsample.conv.bias)  # type: ignore[arg-type]
    torch.nn.init.zeros_(upsample.conv.bias)  # type: ignore[arg-type]
    y = downsample(upsample(x))
    assert y.shape == (1, 3**3, 3, 4, 5)
    torch.testing.assert_close(y, x, msg='Upsample and downsample are not inverses.')
