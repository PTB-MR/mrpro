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

from mrpro.operators import LinearOperator


class WaveletOp(LinearOperator):
    """Wavelet operator class."""

    def __init__(
        self,
        domain_shape: Sequence[int] | None = None,
        dim: Sequence[int] = (-2, -1),
        wavelet_name: str = 'db4',
        level: int | None = None,
    ):
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

            # size of wavelets
            wavelet_length = torch.as_tensor((Wavelet(wavelet_name).dec_len,) * len(domain_shape))

            # calculate shape of wavelet coefficients at each level
            current_shape = torch.as_tensor(domain_shape)

            self.coefficients_shape = []
            self.coefficients_padding = []
            for _ in range(_check_level(domain_shape, wavelet_length, level)):
                # Add padding
                for ind in range(len(current_shape)):
                    padl, padr = _get_pad(current_shape[ind], wavelet_length[ind])
                    current_shape[ind] += padl + padr
                current_shape = torch.floor((current_shape - (wavelet_length - 1) - 1) / 2 + 1).to(dtype=torch.int64)
                self.coefficients_shape.extend([current_shape.clone()] * self.n_wavelet_directions)
                self.coefficients_padding.extend([(padl.clone(), padr.clone())] * self.n_wavelet_directions)

            self.coefficients_shape = self.coefficients_shape[::-1]
            self.coefficients_padding = self.coefficients_padding[::-1]
            self.coefficients_shape.insert(0, self.coefficients_shape[0])  # shape of a/aa/aaa term
            self.coefficients_padding.insert(0, self.coefficients_padding[0])  # padding of a/aa/aaa term

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        if len(self.dim) == 1:
            coefficients_list = wavedec(x, self.wavelet_name, level=self.level)
            coefficients_list = self._format_coeffs_1d(coefficients_list)
        elif len(self.dim) == 2:
            coefficients_list = wavedec2(x, self.wavelet_name, level=self.level)
            coefficients_list = self._format_coeffs_2d(coefficients_list)
        elif len(self.dim) == 3:
            coefficients_list = wavedec3(x, self.wavelet_name, level=self.level)
            coefficients_list = self._format_coeffs_3d(coefficients_list)
        return (self._coeff_to_1d_tensor(coefficients_list),)

    def adjoint(self, coeff_tensor_1d: torch.Tensor) -> tuple[torch.Tensor]:
        coefficients_list = self._1d_tensor_to_coeff(coeff_tensor_1d)
        if len(self.dim) == 1:
            coeffs = self._undo_format_coeffs_1d(coefficients_list)
            data = waverec(coeffs, self.wavelet_name)
        elif len(self.dim) == 2:
            coeffs = self._undo_format_coeffs_2d(coefficients_list)
            data = waverec2(coeffs, self.wavelet_name)
        elif len(self.dim) == 3:
            coeffs = self._undo_format_coeffs_3d(coefficients_list)
            data = waverec3(coeffs, self.wavelet_name)

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
        coeffs_ptwt_format: list = [coefficients.pop(0)]
        for i in range(0, len(coefficients), self.n_wavelet_directions):
            coeffs_ptwt_format.append(
                dict(
                    zip(
                        ['aad', 'ada', 'add', 'data', 'dad', 'dda', 'ddd'],
                        coefficients[i : i + self.n_wavelet_directions],
                        strict=True,
                    )
                )
            )
        return coeffs_ptwt_format

    def _coeff_to_1d_tensor(self, coefficients: list[torch.Tensor]) -> torch.Tensor:
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
        coefficients_tensor_1d = []
        coefficients_tensor_1d.append(coefficients[0].flatten(start_dim=-len(self.dim)))
        for coefficient in coefficients[1:]:
            coefficients_tensor_1d.append(coefficient.flatten(start_dim=-len(self.dim)))
        return torch.cat(coefficients_tensor_1d, dim=-1)

    def _1d_tensor_to_coeff(self, coefficients_tensor_1d: torch.Tensor) -> list[torch.Tensor]:
        """Unstack wavelet coefficients.

        Parameters
        ----------
        coefficients_tensor_1d
            stacked coefficients in tensor [...,coeff]


        Returns
        -------
            wavelet coefficients in the format
            1D: [a, d_n, ..., d_1]
            2D: [aa, ad_n, da_n, dd_n, ..., ad_1, da_1, dd_1]
            3D: [aaa, aad_n, ada_n, add_n, ..., ..., aad_1, ada_1, add_1, ...]
        """
        coefficients = torch.split(
            coefficients_tensor_1d, [int(torch.prod(shape)) for shape in self.coefficients_shape], dim=-1
        )
        return [
            torch.reshape(coeff, coeff.shape[:-1] + tuple(shape))
            for coeff, shape in zip(coefficients, self.coefficients_shape, strict=False)
        ]
