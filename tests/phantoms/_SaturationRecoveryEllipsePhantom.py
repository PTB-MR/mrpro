"""Saturation recovery ellipse phantom for testing."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import torch

from mrpro.data import SpatialDimension
from mrpro.operators import Operator
from mrpro.operators.models import SaturationRecovery
from mrpro.phantoms import EllipsePhantom
from tests._RandomGenerator import RandomGenerator


# TODO: maybe make it possible to use more and different ellipses?
class SaturationRecoveryEllipsePhantom2D:
    def __init__(self, im_shape: SpatialDimension[int], ti: torch.Tensor) -> None:
        self.im_shape = im_shape
        self.ti = ti

    def get_sat_recovery_model(self) -> Operator:
        return SaturationRecovery(self.ti)

    def generate_data(self, noise_std: float = 0.02):
        random_generator = RandomGenerator(seed=0)

        # create ellipse phantom
        phantom = EllipsePhantom()

        # create a simple phantom
        x = phantom.image_space(self.im_shape)

        # create random 2deg polynomial for obatining spatial variability on the
        # ellipses
        def polynom2d(deg, im_shape) -> torch.Tensor:
            """Generate the graph of a random 2d polynomial function."""

            ylin, xlin = torch.linspace(-1, 1, im_shape[-2]), torch.linspace(-1, 1, im_shape[-1])
            yy, xx = torch.meshgrid(ylin, xlin, indexing='ij')

            pol = torch.zeros(im_shape)
            for i in range(deg + 1):
                for j in range(deg + 1 - i):
                    coeff = random_generator.float32_tensor(size=(1,))
                    pol += coeff * yy**i * xx**j

            return pol

        # 2d polynomial of deg=2 for spatial variability
        m0_profile = polynom2d(2, x.shape[-2:])

        # random phase for creating complex-valued information;
        # normalized between [-pi,pi]
        m0_phase = polynom2d(1, x.shape[-2:])
        m0_phase = (m0_phase - m0_phase.min()) / (m0_phase.max() - m0_phase.min())
        m0_phase = torch.pi / 2 * (2 * m0_phase - 1)

        # multiply ellipses image to get spatial variability
        m0_true = (x.abs() * m0_profile).abs() * torch.exp(1j * m0_phase)

        # create a t1 profile; constrain it to be between [eps,5]
        t1_profile = polynom2d(2, x.shape[-2:])
        t1_true = x.abs() * t1_profile
        eps = 0.25  # 250 ms for fat
        t1_true = eps + (5 - eps) * (t1_true - t1_true.min()) / (t1_true.max() - t1_true.min())

        # create model
        q = self.get_sat_recovery_model()

        # create data and corrupt with noise
        sigma = noise_std
        data_clean = q(m0_true, t1_true)
        mu, std = torch.mean(data_clean), torch.std(data_clean)
        data = (data_clean - mu) / std + sigma * random_generator.complex64_tensor(size=data_clean.shape)
        data = mu + std * data

        return m0_true, t1_true, data
