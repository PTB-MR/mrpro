import pytest
import torch
from einops import rearrange

from mrpro.operators.models._FatWater import FatWater
from tests import RandomGenerator


def create_data(n_echoes=2, other=10, coils=5, z=100, y=100, x=100):
    random_generator = RandomGenerator(seed=0)
    water = random_generator.float32_tensor(size=(other, coils, z, y, x), low=1e-10)
    fat = random_generator.float32_tensor(size=(other, coils, z, y, x), low=1e-10)
    phasor = random_generator.float32_tensor(size=(n_echoes, coils, z, y, x), low=1e-10)
    return water, fat, phasor


@pytest.mark.parametrize(
    'n_echoes, other, coils, z, y, x',
    [
        (2, 3, 1, 30, 50, 100),
        (2, 1, 18, 30, 50, 100),
        (3, 3, 1, 30, 50, 100),
        (3, 1, 18, 30, 50, 100),
        (6, 3, 1, 30, 50, 100),
        (6, 1, 18, 30, 50, 100),
    ],
)
def test_fatwater_to_echoes(n_echoes, other, coils, z, y, x):
    """Test for reversing fat water separation.

    Checking that water and fat are combined directly, iff no chemical
    shift.
    """
    # Generate signal model and torch tensor for comparison
    fat_mod = torch.ones((n_echoes, 1, 1, 1, 1))
    model = FatWater(fat_mod)
    water, fat, phasor = create_data(n_echoes, other, coils, z, y, x)
    echoes = model.forward(water, fat, phasor)[0]

    intended_res = (fat[None, :] + water[None, :]) * phasor[:, None]
    intended_res = rearrange(intended_res, 't ... c z y x -> (... t) c z y x')
    assert torch.equal(intended_res, echoes)
