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
from einops import rearrange

from mrpro.operators import SignalModel


class TransientInversionRecovery(SignalModel):
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
    (4) Time between inversion pulse and start of data acquisition, (time_inversion_adc), e.g. due to spoiler gradient
        Mz(t) = M(1 - 2e^(-t * R1)) [M = M0 if no acquisitions before inversion pulse, otherwise it depends on (2)]
    (5) Continuous data acquisition
        Mz(t) = M* - (M + M*)e^(-t * R1*) [M = Mz(time_inversion_adc) using signal equation of (4)]
    (6) Second inversion pulse (inversion_time_points[1]), followed by pause (time_inversion_adc) and next acquisition
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
        inversion_time_points: torch.Tensor,
        time_inversion_adc: float | torch.Tensor,
        first_adc_time_point: float | torch.Tensor | None = None,
    ):
        """Initialize continuous acquisition with inversion pulses.

        Parameters
        ----------
        signal_time_points
            time stamp of each acquisition
        tr
            repetition time
        inversion_time_points
            time stamp of each inversion
        time_inversion_adc
            time between inversion pulse and start of data acquisition
        first_adc_time_point
            time stamp of first acquisition
        """
        super().__init__()
        tr = torch.Tensor(tr)
        time_inversion_adc = torch.Tensor(time_inversion_adc)
        if first_adc_time_point is not None:
            first_adc_time_point = torch.Tensor(first_adc_time_point)
            self.first_adc_time_point = torch.nn.Parameter(
                first_adc_time_point, requires_grad=first_adc_time_point.requires_grad
            )

        self.signal_time_points = torch.nn.Parameter(signal_time_points, requires_grad=signal_time_points.requires_grad)
        self.tr = torch.nn.Parameter(tr, requires_grad=tr.requires_grad)
        self.time_inversion_adc = torch.nn.Parameter(time_inversion_adc, requires_grad=time_inversion_adc.requires_grad)

        self.index_before_first_inversion = torch.where(signal_time_points < inversion_time_points[0])[0]
        if len(self.index_before_first_inversion) > 0 and self.first_adc_time_point is None:
            raise ValueError(
                'If data has been acquired before the first inversion pulse,',
                'the start of the acquisitions first_adc_time_point has to be defined.',
            )

        # Verify that no acquisition occurred before start of acquisition
        if self.first_adc_time_point is not None and any(signal_time_points < first_adc_time_point):
            raise ValueError('Acquisitions detected before start of acquisition.')

        # Add final "virtual" inversion pulse to allow for easier looping later
        inversion_time_points = torch.concat((inversion_time_points, (signal_time_points.max() + 1).reshape(1)), dim=0)
        self.inversion_time_points = torch.nn.Parameter(
            inversion_time_points, requires_grad=inversion_time_points.requires_grad
        )

        # Get index of data points between different inversion pulses
        self.index_between_inversions = []
        for inversion_index in range(len(self.inversion_time_points) - 1):
            # Get points between ind and ind+1 inversion pulse
            index_of_time_points = torch.where(
                (signal_time_points < self.inversion_time_points[inversion_index + 1])
                & (signal_time_points >= self.inversion_time_points[inversion_index])
            )[0]

            # Verify that no points lie between the inversion pulse and the first acquisition
            points_before_adc = torch.where(
                signal_time_points[index_of_time_points]
                < (inversion_time_points[inversion_index] + self.time_inversion_adc)
            )[0]
            if len(points_before_adc) > 0:
                raise ValueError('No data points should lie between inversion pulse and first acquisition')

            self.index_between_inversions.append(index_of_time_points)

    def forward(self, m0: torch.Tensor, t1: torch.Tensor, alpha: torch.Tensor):
        """Apply Transient Inversion Recovery signal model.

        Parameters
        ----------
        m0
            equilibrium signal / proton density
        t1
            longitudinal relaxation time T1
        alpha
            flip angle

        Returns
        -------
            signal with dimensions ((... times), coils, z, y, x)
        """
        t1 = torch.where(t1 == 0, 1e-10, t1)
        t1_star = 1 / (1 / t1 - torch.log(torch.cos(torch.deg2rad(alpha))) / self.tr)
        m0_star = m0 * t1_star / t1

        signal = torch.zeros((len(self.signal_time_points), *m0.shape), dtype=torch.complex64)

        if len(self.index_before_first_inversion) > 0:
            # Time relative to first acquisition
            time_since_fist_acquisition = (
                self.signal_time_points[self.index_before_first_inversion] - self.first_adc_time_point
            )[(...,) + (None,) * (m0.ndim)]

            signal[self.index_before_first_inversion, ...] = (
                m0_star + (m0 - m0_star) * torch.exp(-time_since_fist_acquisition / t1_star)
            ).to(dtype=torch.complex64)

        # Signal just before first inversion pulse
        if self.first_adc_time_point is not None and self.first_adc_time_point < self.inversion_time_points[0]:
            time = self.inversion_time_points[0] - self.first_adc_time_point
            m0_plus = m0_star + (m0 - m0_star) * torch.exp(-time / t1_star)
        else:
            m0_plus = m0

        for inversion_index, index_of_time_points in enumerate(self.index_between_inversions):
            # Calculate signal at the beginning of acquisition after the inversion pulse
            m0_tau = -m0_plus * (1 - 2 * torch.exp(-self.time_inversion_adc / t1))

            if len(index_of_time_points) > 0:
                # Inversion times relative to current inversion pulse + time between inversion pulse and start of
                # acquisition
                inversion_time = (
                    self.signal_time_points[index_of_time_points]
                    - self.inversion_time_points[inversion_index]
                    - self.time_inversion_adc
                )[(...,) + (None,) * (m0.ndim)]

                signal[index_of_time_points, ...] = (
                    m0_star - (m0_tau + m0_star) * torch.exp(-inversion_time / t1_star)
                ).to(dtype=torch.complex64)

            # Signal at the beginning of the next inversion pulse
            time = (
                self.inversion_time_points[inversion_index + 1]
                - self.inversion_time_points[inversion_index]
                - self.time_inversion_adc
            )
            m0_plus = m0_star - (m0_tau + m0_star) * torch.exp(-time / t1_star)

        return (rearrange(signal, 't ... c z y x -> (... t) c z y x'),)
