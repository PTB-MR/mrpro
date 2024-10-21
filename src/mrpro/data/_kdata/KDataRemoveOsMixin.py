"""Remove oversampling along readout dimension."""

from copy import deepcopy

import torch
from typing_extensions import Self

from mrpro.data._kdata.KDataProtocol import _KDataProtocol
from mrpro.data.KTrajectory import KTrajectory


class KDataRemoveOsMixin(_KDataProtocol):
    """Remove oversampling along readout dimension."""

    def remove_readout_os(self: Self) -> Self:
        """Remove any oversampling along the readout (k0) direction [GAD]_.

        Returns a copy of the data.

        Parameters
        ----------
        kdata
            K-space data

        Returns
        -------
            Copy of K-space data with oversampling removed.

        Raises
        ------
        ValueError
            If the recon matrix along x is larger than the encoding matrix along x.

        References
        ----------
        .. [GAD] Gadgetron https://github.com/gadgetron/gadgetron-python
        """
        from mrpro.operators.FastFourierOp import FastFourierOp

        # Ratio of k0/x between encoded and recon space
        x_ratio = self.header.recon_matrix.x / self.header.encoding_matrix.x
        if x_ratio == 1:
            # If the encoded and recon space is the same we don't have to do anything
            return self
        elif x_ratio > 1:
            raise ValueError('Recon matrix along x should be equal or larger than encoding matrix along x.')

        # Starting and end point of image after removing oversampling
        start_cropped_readout = (self.header.encoding_matrix.x - self.header.recon_matrix.x) // 2
        end_cropped_readout = start_cropped_readout + self.header.recon_matrix.x

        def crop_readout(data_to_crop: torch.Tensor) -> torch.Tensor:
            # returns a cropped copy
            return data_to_crop[..., start_cropped_readout:end_cropped_readout].clone()

        # Transform to image space along readout, crop to reconstruction matrix size and transform back
        fourier_k0_op = FastFourierOp(dim=(-1,))
        (cropped_data,) = fourier_k0_op(crop_readout(*fourier_k0_op.H(self.data)))

        # Adapt trajectory
        ks = [self.traj.kz, self.traj.ky, self.traj.kx]
        # only cropped ks that are not broadcasted/singleton along k0
        cropped_ks = [crop_readout(k) if k.shape[-1] > 1 else k.clone() for k in ks]
        cropped_traj = KTrajectory(cropped_ks[0], cropped_ks[1], cropped_ks[2])

        # Adapt header parameters
        header = deepcopy(self.header)
        header.acq_info.center_sample -= start_cropped_readout
        header.acq_info.number_of_samples[:] = cropped_data.shape[-1]
        header.encoding_matrix.x = cropped_data.shape[-1]

        header.acq_info.discard_post = (header.acq_info.discard_post * x_ratio).to(torch.int32)
        header.acq_info.discard_pre = (header.acq_info.discard_pre * x_ratio).to(torch.int32)

        return type(self)(header, cropped_data, cropped_traj)
