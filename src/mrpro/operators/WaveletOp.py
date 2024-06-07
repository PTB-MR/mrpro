"""Wavelet operator."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
from ptwt.conv_transform import _get_pad
from ptwt.conv_transform import wavedec
from ptwt.conv_transform import waverec
from ptwt.conv_transform_2 import wavedec2
from ptwt.conv_transform_2 import waverec2
from ptwt.conv_transform_3 import wavedec3
from ptwt.conv_transform_3 import waverec3
from pywt import Wavelet
from pywt._multilevel import _check_level

from mrpro.operators.LinearOperator import LinearOperator


class WaveletOp(LinearOperator):
    """Wavelet operator class."""

    def __init__(
        self,
        domain_shape: Sequence[int] | None = None,
        dim: tuple[int] | tuple[int, int] | tuple[int, int, int] = (-2, -1),
        wavelet_name: str = 'db4',
        level: int | None = None,
    ):
        """Wavelet operator.

        Parameters
        ----------
        domain_shape
            Shape of domain where wavelets are calculated. If set to None the shape is taken from the input of the
            forward operator. The adjoint operator will raise an error.
        dim
            Dimensions (axes) where wavelets are calculated
        wavelet_name
            Name of wavelets
        level
            Highest wavelet level. If set to None, the highest possible level is calculated based on the domain shape.

        Raises
        ------
        ValueError
            If wavelets are calculated for more than three dimensions.
        ValueError
            If wavelet dimensions and domain shape do not match.
        NotImplementedError
            If the domain shape is odd. Adjoint will lead to the wrong domain shape.
        """
        super().__init__()
        self.domain_shape = domain_shape
        self.wavelet_name = wavelet_name
        self.dim = dim
        self.level = level

        # number of wavelet directions
        if len(dim) == 1:
            self.n_wavelet_directions = 1
        elif len(dim) == 2:
            self.n_wavelet_directions = 3
        elif len(dim) == 3:
            self.n_wavelet_directions = 7
        else:
            raise ValueError('Only 1D, 2D and 3D wavelet transforms are supported.')

        if domain_shape is not None:
            if len(dim) != len(domain_shape):
                raise ValueError(
                    'Number of dimensions along which the wavelet transform should be calculated needs to',
                    'be the same as the domain shape',
                )

            if any(d % 2 for d in domain_shape):
                raise NotImplementedError('ptwt only supports wavelet transforms for even number of samples.')

            # size of wavelets
            wavelet_length = torch.as_tensor((Wavelet(wavelet_name).dec_len,) * len(domain_shape))

            # calculate shape of wavelet coefficients at each level
            current_shape = torch.as_tensor(domain_shape)

            if _check_level(domain_shape, wavelet_length, level) == 0:
                self.coefficients_shape = [current_shape]
                self.coefficients_padding = [(0, 0)] * len(current_shape)
            else:
                self.coefficients_shape = []
                self.coefficients_padding = []
                for _ in range(_check_level(domain_shape, wavelet_length, level)):
                    # Add padding
                    for ind in range(len(current_shape)):
                        padl, padr = _get_pad(current_shape[ind], wavelet_length[ind])
                        current_shape[ind] += padl + padr
                    current_shape = torch.floor((current_shape - (wavelet_length - 1) - 1) / 2 + 1).to(
                        dtype=torch.int64
                    )
                    self.coefficients_shape.extend([current_shape.clone()] * self.n_wavelet_directions)
                    self.coefficients_padding.extend([(padl.clone(), padr.clone())] * self.n_wavelet_directions)

                self.coefficients_shape = self.coefficients_shape[::-1]
                self.coefficients_padding = self.coefficients_padding[::-1]
                self.coefficients_shape.insert(0, self.coefficients_shape[0])  # shape of a/aa/aaa term
                self.coefficients_padding.insert(0, self.coefficients_padding[0])  # padding of a/aa/aaa term

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Calculate wavelet coefficients from (image) data.

        Parameters
        ----------
        x
            (Image) data

        Returns
        -------
            Wavelet coefficients stacked along one dimension

        Raises
        ------
        ValueError
            If the dimensions along which wavelets are to be calculated are not unique.
        """
        # normalize axes to allow negative indexing in input
        dim = tuple(d % x.ndim for d in self.dim)
        if len(dim) != len(set(dim)):
            raise ValueError(f'Axis must be unique. Normalized axis are {dim}')

        # swapping the last axes and the axes to calculate wavelets of
        x = torch.moveaxis(x, dim, list(range(-len(self.dim), 0)))

        # the ptwt functions work only for real data, thus we handle complex inputs as an additional channel
        x_real = torch.view_as_real(x).moveaxis(-1, 0) if x.is_complex() else x

        if len(self.dim) == 1:
            coeffs1 = wavedec(x_real, self.wavelet_name, level=self.level, mode='zero', axis=-1)
            coefficients_list = self._format_coeffs_1d(coeffs1)
        elif len(self.dim) == 2:
            coeffs2 = wavedec2(x_real, self.wavelet_name, level=self.level, mode='zero', axes=(-2, -1))
            coefficients_list = self._format_coeffs_2d(coeffs2)
        elif len(self.dim) == 3:
            coeffs3 = wavedec3(x_real, self.wavelet_name, level=self.level, mode='zero', axes=(-3, -2, -1))
            coefficients_list = self._format_coeffs_3d(coeffs3)

        # stack multi-resolution wavelets along single dimension
        coefficients_stack = self._coeff_to_stacked_tensor(coefficients_list)
        coefficients_stack = (
            torch.view_as_complex(coefficients_stack.moveaxis(0, -1).contiguous())
            if x.is_complex()
            else coefficients_stack
        )

        # move stacked coefficients to first wavelet dimension
        return (torch.moveaxis(coefficients_stack, -1, min(dim)),)

    def adjoint(self, coefficients_stack: torch.Tensor) -> tuple[torch.Tensor]:
        """Transform wavelet coefficients to (image) data.

        Parameters
        ----------
        coefficients_stack
            Wavelet coefficients stacked along one dimension

        Returns
        -------
            (Image) data

        Raises
        ------
        ValueError
            If the domain_shape is not defined.
        ValueError
            If the dimensions along which wavelets are to be calculated are not unique.
        """
        if self.domain_shape is None:
            raise ValueError('Adjoint requires to define the domain_shape in init()')

        # normalize axes to allow negative indexing in input
        dim = tuple(d % (coefficients_stack.ndim + len(self.dim) - 1) for d in self.dim)
        if len(dim) != len(set(dim)):
            raise ValueError(f'Axis must be unique. Normalized axis are {dim}')

        coefficients_stack = torch.moveaxis(coefficients_stack, min(dim), -1)

        # the ptwt functions work only for real data, thus we handle complex inputs as an additional channel
        coefficients_stack_real = (
            torch.view_as_real(coefficients_stack).moveaxis(-1, 0)
            if coefficients_stack.is_complex()
            else coefficients_stack
        )

        coefficients_list = self._stacked_tensor_to_coeff(coefficients_stack_real)

        if len(self.dim) == 1:
            coeffs1 = self._undo_format_coeffs_1d(coefficients_list)
            data = waverec(coeffs1, self.wavelet_name, axis=-1)
        elif len(self.dim) == 2:
            coeffs2 = self._undo_format_coeffs_2d(coefficients_list)
            data = waverec2(coeffs2, self.wavelet_name, axes=(-2, -1))
        elif len(self.dim) == 3:
            coeffs3 = self._undo_format_coeffs_3d(coefficients_list)
            data = waverec3(coeffs3, self.wavelet_name, axes=(-3, -2, -1))

        # if we deal with complex coefficients, we also return complex data
        if coefficients_stack.is_complex():
            data = torch.view_as_complex(data.moveaxis(0, -1).contiguous())

        # undo swapping of axes
        data = torch.moveaxis(data, list(range(-len(self.dim), 0)), dim)

        return (data,)

    def _format_coeffs_1d(self, coefficients: list[torch.Tensor]) -> list[torch.Tensor]:
        """Format 1D wavelet coefficients to MRpro format.

        At the moment, this function just returns the input coefficients as is:
        [a, d_n, ..., d_1]
        """
        return coefficients

    def _undo_format_coeffs_1d(self, coefficients: list[torch.Tensor]) -> list[torch.Tensor]:
        """Undo format 1D wavelet coefficients to MRpro format.

        At the moment, this function just returns the input coefficients as is:
        [a, d_n, ..., d_1]
        """
        return coefficients

    def _format_coeffs_2d(
        self,
        coefficients: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> list[torch.Tensor]:
        """Format 2D wavelet coefficients to MRpro format.

        Converts from   [aa, (ad_n, da_n, dd_n), ..., (ad_1, da_1, dd_1)]
        to              [aa, ad_n, da_n, dd_n, ..., ad_1, da_1, dd_1]
        """
        coeffs_mrpro_format: list = [coefficients[0]]
        for c_tuple in coefficients[1:]:
            coeffs_mrpro_format.extend(c_tuple)
        return coeffs_mrpro_format

    def _undo_format_coeffs_2d(
        self,
        coefficients: list[torch.Tensor],
    ) -> list[torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Undo format 2D wavelet coefficients to MRpro format.

        Converts from   [aa, ad_n, da_n, dd_n, ..., ad_1, da_1, dd_1]
        to              [aa, (ad_n, da_n, dd_n), ..., (ad_1, da_1, dd_1)]
        """
        coeffs_ptwt_format: list = [coefficients[0]]
        for i in range(1, len(coefficients), self.n_wavelet_directions):
            coeffs_ptwt_format.append(tuple(coefficients[i : i + self.n_wavelet_directions]))
        return coeffs_ptwt_format

    def _format_coeffs_3d(self, coefficients: list[torch.Tensor | dict[str, torch.Tensor]]) -> list[torch.Tensor]:
        """Format 3D wavelet coefficients to MRpro format.

        Converts from   [aaa, {aad_n, ada_n, add_n, ...}, ..., {aad_1, ada_1, add_1, ...}]
        to              [aaa, aad_n, ada_n, add_n, ..., ..., aad_1, ada_1, add_1, ...]
        """
        coeffs_mrpro_format: list = [coefficients[0]]
        for c_dict in coefficients[1:]:
            coeffs_mrpro_format.extend(c_dict.values())
        return coeffs_mrpro_format

    def _undo_format_coeffs_3d(
        self,
        coefficients: list[torch.Tensor],
    ) -> list[torch.Tensor | dict[str, torch.Tensor]]:
        """Undo format 3D wavelet coefficients to MRpro format.

        Converts from   [aaa, aad_n, ada_n, add_n, ..., ..., aad_1, ada_1, add_1, ...]
        to              [aaa, {aad_n, ada_n, add_n, ...}, ..., {aad_1, ada_1, add_1, ...}]
        """
        coeffs_ptwt_format: list = [coefficients[0]]
        for i in range(1, len(coefficients), self.n_wavelet_directions):
            coeffs_ptwt_format.append(
                dict(
                    zip(
                        ['aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd'],
                        coefficients[i : i + self.n_wavelet_directions],
                        strict=True,
                    )
                )
            )
        return coeffs_ptwt_format

    def _coeff_to_stacked_tensor(self, coefficients: list[torch.Tensor]) -> torch.Tensor:
        """Stack wavelet coefficients into 1D tensor.

        During the calculation of the of the wavelet coefficient ptwt uses padding. To ensure the wavelet operator is
        an isometry, cropping is needed.

        Parameters
        ----------
        coefficients
            wavelet coefficients in the format
            1D: [a, d_n, ..., d_1]
            2D: [aa, ad_n, da_n, dd_n, ..., ad_1, da_1, dd_1]
            3D: [aaa, aad_n, ada_n, add_n, ..., ..., aad_1, ada_1, add_1, ...]

        Returns
        -------
            stacked coefficients in tensor [...,coeff]
        """
        return torch.cat(
            [coeff.flatten(start_dim=-len(self.dim)) for coeff in coefficients] if len(self.dim) > 1 else coefficients,
            dim=-1,
        )

    def _stacked_tensor_to_coeff(self, coefficients_stack: torch.Tensor) -> list[torch.Tensor]:
        """Unstack wavelet coefficients.

        Parameters
        ----------
        coefficients_stack
            stacked coefficients in tensor [...,coeff]


        Returns
        -------
            wavelet coefficients in the format
            1D: [a, d_n, ..., d_1]
            2D: [aa, ad_n, da_n, dd_n, ..., ad_1, da_1, dd_1]
            3D: [aaa, aad_n, ada_n, add_n, ..., ..., aad_1, ada_1, add_1, ...]
        """
        coefficients = torch.split(
            coefficients_stack, [int(torch.prod(shape)) for shape in self.coefficients_shape], dim=-1
        )
        return [
            torch.reshape(coeff, coeff.shape[:-1] + tuple(shape))
            for coeff, shape in zip(coefficients, self.coefficients_shape, strict=False)
        ]