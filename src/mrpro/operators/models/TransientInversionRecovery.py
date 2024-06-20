"""Transient steady state signal model with inversion pulse."""

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

from collections.abc import Sequence

import torch

from mrpro.operators.SignalModel import SignalModel


class TransientInversionRecovery(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Transient steady state signal model with inversion pulses.

    This signal model describes the behavior of the longitudinal magnetisation during continuous acquisition with
    multiple inversion pulses. All the timings (signal_time_points, inversion_time_points, first_adc_time_point)
    have to be given relative to the same clock. Values can be given in s, ms... but have to have all the same units.

    Overview of the sequence
    M0 equilibrium magnetisation
    Mz measured magnetisation
    M* steady-state magnetisation


    (1) Start of data acquisition (first_adc_time_point) (optional)
    (2) Data acquisition before first inversion pulse (optional)
        Mz(t) = M* + (M0 - M*)e^(-t * R1*)
    (3) First inversion pulse (inversion_time_points[0])
    (4) Time between inversion pulse and start of data acquisition, (delay_inversion_adc), e.g. due to spoiler gradient
        Mz(t) = M(1 - 2e^(-t * R1)) [M = M0 if no acquisitions before inversion pulse, otherwise it depends on (2)]
    (5) Continuous data acquisition
        Mz(t) = M* - (M + M*)e^(-t * R1*) [M = Mz(delay_inversion_adc) using signal equation of (4)]
    (6) Second inversion pulse (inversion_time_points[1]), followed by pause (delay_inversion_adc) and next acquisition
        block

    (1)      (2)       (3)   (4)             (5)       (6)
    |-----------------|INV|-------||------------------|INV|-------||----------------....

    The model does not have to be evaluated for all the time points obtained during data acquisition. By defining the
    start of the data acquisition, the time points of the inversion pulses and the gap between inversion pulses and the
    first data acquisition, the sequence is fully described. This allows for the calculation of a signal from any
    arbitrary subset of the data acquisition.


    More information on this signal model can be found in:
    Becker, K. M., Schulz-Menger, J., Schaeffter, T. & Kolbitsch, C.
    Simultaneous high-resolution cardiac T1 mapping and cine imaging using model-based iterative image reconstruction.
    Magn. Reson. Med. 81, 1080-1091 (2019). http://doi.wiley.com/10.1002/mrm.27474
    """

    def __init__(
        self,
        signal_time_points: torch.Tensor,
        tr: float | torch.Tensor,
        inversion_time_points: float | torch.Tensor,
        delay_inversion_adc: float | torch.Tensor = 0.0,
        first_adc_time_point: float | torch.Tensor | None = None,
    ):
        """Initialize continuous acquisition with inversion pulses.

        Parameters
        ----------
        signal_time_points
            time stamp of each acquisition
            with shape (time, ...)
        tr
            repetition time
        inversion_time_points
            time stamp of each inversion
            with shape (n_inversions, ...)
        delay_inversion_adc
            time between inversion pulse and start of data acquisition
        first_adc_time_point
            time stamp of first acquisition
        """
        super().__init__()
        tr = torch.as_tensor(tr)
        inversion_time_points = torch.atleast_1d(torch.as_tensor(inversion_time_points))
        delay_inversion_adc = torch.as_tensor(delay_inversion_adc)
        first_adc_time_point = torch.as_tensor(first_adc_time_point) if first_adc_time_point is not None else None

        # signal_time_points, inversion_time_points, tr and delay_inversion_adc have to be broadcastable
        # first_adc_time_point too, if not None
        # multiple inversion times are possible for one signal
        input_parameter_shape = [tr.shape, inversion_time_points[0, ...].shape, delay_inversion_adc.shape]
        input_parameter_names = ['tr', 'inversion_time_points', 'delay_inversion_adc']
        if first_adc_time_point is not None:
            input_parameter_shape.append(first_adc_time_point.shape)
            input_parameter_names.append('first_adc_time_point')
        for par_shape, par_name in zip(input_parameter_shape, input_parameter_names, strict=False):
            try:
                torch.broadcast_shapes(par_shape, signal_time_points[0, ...].shape)
            except RuntimeError:
                # Not broadcastable
                raise ValueError(
                    f'Broadcasted shape of {par_name} does not match: {par_shape} vs {signal_time_points[0,...].shape}.'
                ) from None

        self.signal_time_points = torch.nn.Parameter(signal_time_points, requires_grad=signal_time_points.requires_grad)
        self.tr = torch.nn.Parameter(tr, requires_grad=tr.requires_grad)
        self.inversion_time_points = torch.nn.Parameter(
            inversion_time_points, requires_grad=inversion_time_points.requires_grad
        )
        self.delay_inversion_adc = torch.nn.Parameter(
            delay_inversion_adc, requires_grad=delay_inversion_adc.requires_grad
        )
        self.first_adc_time_point = (
            torch.nn.Parameter(first_adc_time_point, requires_grad=first_adc_time_point.requires_grad)
            if first_adc_time_point is not None
            else None
        )

    @staticmethod
    def _forward_single_voxel(
        m0: torch.Tensor,
        t1: torch.Tensor,
        alpha: torch.Tensor,
        signal_time_points: torch.Tensor,
        tr: torch.Tensor,
        inversion_time_points: torch.Tensor,
        delay_inversion_adc: torch.Tensor,
        first_adc_time_point: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Transient Inversion Recovery to single voxel.

        Parameters
        ----------
        m0
            equilibrium signal / proton density
        t1
            longitudinal relaxation time T1
        alpha
            flip angle
        signal_time_points
            time stamp of each acquisition
        tr
            repetition time
        inversion_time_points
            time stamp of each inversion
        delay_inversion_adc
            time between inversion pulse and start of data acquisition
        first_adc_time_point
            time stamp of first acquisition


        Returns
        -------
            1D signal
        """
        # Need to sort timings
        signal_time_points, signal_index = torch.sort(signal_time_points, stable=True)
        inversion_time_points, _ = torch.sort(inversion_time_points)

        t1 = torch.where(t1 == 0, 1e-10, t1)
        t1_star = 1 / (1 / t1 - torch.log(torch.cos(alpha)) / tr)
        m0_star = m0 * t1_star / t1

        signal = []

        # Signal before first inversion pulse
        index_before_first_inversion = torch.where(signal_time_points < inversion_time_points[0])[0]
        if len(index_before_first_inversion) > 0:
            if first_adc_time_point is None:
                raise ValueError(
                    'If data has been acquired before the first inversion pulse,',
                    'the start of the acquisitions first_adc_time_point has to be defined.',
                )

            # Time relative to first acquisition
            time_since_fist_acquisition = signal_time_points[index_before_first_inversion] - first_adc_time_point

            signal.append(
                (m0_star + (m0 - m0_star) * torch.exp(-time_since_fist_acquisition / t1_star)).to(dtype=torch.complex64)
            )

        # Verify that no acquisition occurred before start of acquisition
        if first_adc_time_point is not None and any(signal_time_points < first_adc_time_point):
            raise ValueError('Acquisitions detected before start of acquisition.')

        # Signal just before first inversion pulse
        if first_adc_time_point is not None and first_adc_time_point < inversion_time_points[0]:
            time = inversion_time_points[0] - first_adc_time_point
            m0_plus = m0_star + (m0 - m0_star) * torch.exp(-time / t1_star)
        else:
            m0_plus = m0

        # Add final "virtual" inversion pulse to allow for easier looping later
        inversion_time_points = torch.concat((inversion_time_points, (signal_time_points[-1] + 1).reshape(1)), dim=0)
        for inversion_index in range(len(inversion_time_points) - 1):
            # Calculate signal at the beginning of acquisition after the inversion pulse
            m0_tau = -m0_plus * (1 - 2 * torch.exp(-delay_inversion_adc / t1))

            # Get points between ind and ind+1 inversion pulse
            index_of_time_points = torch.where(
                (signal_time_points < inversion_time_points[inversion_index + 1])
                & (signal_time_points >= inversion_time_points[inversion_index])
            )[0]

            # Verify that no points lie between the inversion pulse and the first acquisition
            points_before_adc = torch.where(
                signal_time_points[index_of_time_points]
                < (inversion_time_points[inversion_index] + delay_inversion_adc)
            )[0]
            if len(points_before_adc) > 0:
                raise ValueError('No data points should lie between inversion pulse and first acquisition')

            if len(index_of_time_points) > 0:
                # Inversion times relative to current inversion pulse + time between inversion pulse and start of
                # acquisition
                inversion_time = (
                    signal_time_points[index_of_time_points]
                    - inversion_time_points[inversion_index]
                    - delay_inversion_adc
                )

                signal.append(
                    (m0_star - (m0_tau + m0_star) * torch.exp(-inversion_time / t1_star)).to(dtype=torch.complex64)
                )

                # Signal at the beginning of the next inversion pulse
                time = (
                    inversion_time_points[inversion_index + 1]
                    - inversion_time_points[inversion_index]
                    - delay_inversion_adc
                )
                m0_plus = m0_star - (m0_tau + m0_star) * torch.exp(-time / t1_star)

        return torch.cat(signal)[signal_index]

    def forward(self, m0: torch.Tensor, t1: torch.Tensor, alpha: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply Transient Inversion Recovery signal model.

        Parameters
        ----------
        m0
            equilibrium signal / proton density
            with shape (... other, coils, z, y, x)
        t1
            longitudinal relaxation time T1
            with shape (... other, coils, z, y, x)
        alpha
            flip angle
            with shape (... other, coils, z, y, x)

        Returns
        -------
            signal
            with shape (time ... other, coils, z, y, x)
        """

        def parameter_for_vmap(
            parameter: torch.Tensor, shape_of_input_tensor: Sequence[int], n_additional_parameter_dims: int = 0
        ):
            # First we ensure that there are no trailing single dimensions, i.e [4,1,2,1,1] -> [4,1,2]
            # This ensures a parameter of shape [6,1,1,1] is broadcasted in the same way as [6,]
            if parameter.ndim > 1:
                last_non_single_dim = [index for index, dim in enumerate(parameter.shape) if dim != 1][-1]
                parameter = parameter.squeeze(dim=tuple(range(last_non_single_dim + 1, parameter.ndim)))

            # If the parameter values are the same for all m0/t1/alpha entries then we don't have to do anything.
            # Otherwise we need to broadcast the parameters
            delta_ndim = len(shape_of_input_tensor) - (parameter.ndim - n_additional_parameter_dims)
            if delta_ndim == len(shape_of_input_tensor):  # parameter can be fully broadcasted
                return parameter, None
            else:
                return torch.broadcast_to(
                    parameter[..., *[None] * (delta_ndim)], (*parameter.shape, *shape_of_input_tensor[-delta_ndim:])
                ), -1

        m0_shape = m0.shape
        signal_time_points, signal_time_points_vmap_indim = parameter_for_vmap(self.signal_time_points, m0_shape, 1)
        tr, tr_vmap_indim = parameter_for_vmap(self.tr, m0_shape)
        inversion_time_points, inversion_time_points_vmap_indim = parameter_for_vmap(
            self.inversion_time_points, m0_shape, 1
        )
        delay_inversion_adc, delay_inversion_adc_vmap_indim = parameter_for_vmap(self.delay_inversion_adc, m0_shape)
        first_adc_time_point, first_adc_time_point_vmap_indim = parameter_for_vmap(self.first_adc_time_point, m0_shape)

        vmap_forward = torch.func.vmap(
            self._forward_single_voxel,
            in_dims=(
                -1,
                -1,
                -1,
                signal_time_points_vmap_indim,
                tr_vmap_indim,
                inversion_time_points_vmap_indim,
                delay_inversion_adc_vmap_indim,
                first_adc_time_point_vmap_indim,
            ),
            out_dims=-1,
        )
        return (
            torch.reshape(
                vmap_forward(
                    m0.flatten(),
                    t1.flatten(),
                    alpha.flatten(),
                    signal_time_points.flatten(start_dim=1),
                    tr.flatten(),
                    inversion_time_points.flatten(start_dim=1),
                    delay_inversion_adc.flatten(),
                    first_adc_time_point.flatten(),
                ),
                [-1, *m0.shape],
            ),
        )
