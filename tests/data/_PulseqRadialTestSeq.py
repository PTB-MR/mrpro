"""Create Pulseq test file with radial trajectory."""

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

import pypulseq
import torch
from mrpro.data import KTrajectory


class PulseqRadialTestSeq:
    def __init__(self, seq_filename: str, n_x=256, n_spokes=10):
        """A radial 2D trajectory in Pulseq.

        Please note: this is not a working sequence as delays, spoiler, etc are nonsense.

        Parameters
        ----------
        seq_filename
            target filename
        n_x
            number of frequency encoding points
        n_spokes
            number of spokes
        """
        seq = pypulseq.Sequence()
        fov = 200e-3

        delta_angle = torch.pi / n_spokes
        delta_k = 1 / fov

        system = pypulseq.Opts()
        rf, gz, _ = pypulseq.make_sinc_pulse(flip_angle=0.1, slice_thickness=1e-3, system=system, return_gz=True)
        gx = pypulseq.make_trapezoid(channel='x', flat_area=n_x * delta_k, flat_time=2e-3, system=system)
        adc = pypulseq.make_adc(num_samples=n_x, duration=gx.flat_time, delay=gx.rise_time, system=system)
        gx_pre = pypulseq.make_trapezoid(channel='x', area=-gx.area / 2 - delta_k / 2, duration=2e-3, system=system)
        gz_reph = pypulseq.make_trapezoid(channel='z', area=-gz.area / 2, duration=2e-3, system=system)

        for spoke in range(n_spokes):
            angle = delta_angle * spoke
            seq.add_block(rf, gz)
            seq.add_block(*pypulseq.rotate(gx_pre, gz_reph, angle=angle, axis='z'))
            seq.add_block(pypulseq.make_delay(10e-3))
            seq.add_block(*pypulseq.rotate(gx, adc, angle=angle, axis='z'))
            seq.add_block(pypulseq.make_delay(100e-3))

        seq.write(str(seq_filename))

        self.n_x = n_x
        self.n_spokes = n_spokes
        self.seq = seq
        self.seq_filename = seq_filename

        kz = torch.zeros(1, 1, n_spokes, n_x)
        angle = torch.pi / n_spokes * torch.arange(n_spokes)
        k0 = delta_k * torch.linspace(-n_x / 2, n_x / 2 - 1, n_x)
        kx = (torch.cos(angle)[:, None] * k0)[None, None, ...]
        ky = (torch.sin(angle)[:, None] * k0)[None, None, ...]

        self.traj_analytical = KTrajectory(kz, ky, kx)
