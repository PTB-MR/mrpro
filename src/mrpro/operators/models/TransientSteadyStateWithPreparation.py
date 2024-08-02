"""Transient steady state signal models."""

import torch

from mrpro.operators.SignalModel import SignalModel


class TransientSteadyStateWithPreparation(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Signal model for transient steady state.

    This signal model describes the behavior of the longitudinal magnetization during continuous acquisition after
    a preparation pulse. The effect of the preparation pulse is modelled by a scaling factor applied to the
    equilibrium magnetization. A delay after the preparation pulse can be defined. During this time T1 relaxation to M0
    occurs. Data acquisition starts after this delay. Perfect spoiling is assumed and hence T2 effects are not
    considered in the model. In addition this model assumes TR << T1 and TR << T1* (see definition below) [1]_ [2]_.

    Let's assume we want to describe a continuous acquisition after an inversion pulse, then we have three parts:
    [Part A: 180° inversion pulse][Part B: spoiler gradient][Part C: Continuous data acquisition]

    Part A: The 180° pulse leads to an inversion of the equilibrium magnetization (M0) to -M0. This can be described by
            setting the scaling factor m0_scaling_preparation to -1

    Part B: Commonly after an inversion pulse a strong spoiler gradient is played out to compensate for non-perfect
            inversion. During this time the magnetization follows Mz(t) the signal model:
                    Mz(t) = M0 + (m0_scaling_preparation*M0 - M0)e^(-t / T1)

    Part C: After the spoiler gradient the data acquisition starts and the magnetization Mz(t) can be described by the
            signal model:
                    Mz(t) = M0* + (M0_init - M0*)e^(-t / T1*)
            where the initial magnetization is
                    M0_init = M0 + (m0_scaling_preparation*M0 - M0)e^(-delay_after_preparation / T1)
            the effective longitudinal relaxation time is
                    T1* = 1/(1/T1 - 1/repetition_time ln(cos(flip_angle)))
            and the steady-state magnetization is
                    M0* = M0 T1* / T1

    References
    ----------
    .. [1] Deichmann R, Haase A (1992) Quantification of T1 values by SNAPSHOT-FLASH NMR imaging. J. Magn. Reson. 612
        http://doi.org/10.1016/0022-2364(92)90347-A
    .. [2] Look D, Locker R (1970) Time Saving in Measurement of NMR and EPR Relaxation Times. Rev. Sci. Instrum 41
        https://doi.org/10.1063/1.1684482
    """

    def __init__(
        self,
        sampling_time: float | torch.Tensor,
        repetition_time: float | torch.Tensor,
        m0_scaling_preparation: float | torch.Tensor = 1.0,
        delay_after_preparation: float | torch.Tensor = 0.0,
    ):
        """Initialize transient steady state signal model.

        `repetition_time`, `m0_scaling_preparation` and `delay_after_preparation` can vary for each voxel and will be
        broadcasted starting from the front (i.e. from the other dimension).

        Parameters
        ----------
        sampling_time
            Time points when model is evaluated. A sampling_time of 0 describes the first acquired data point after the
            inversion pulse and spoiler gradients. To take the T1 relaxation during the delay between inversion pulse
            and start of data acquisition into account, set the delay_after_preparation > 0.
            with shape (time, ...)
        repetition_time
            repetition time
        m0_scaling_preparation
            Scaling of the equilibrium magnetization due to the preparation pulse before the data acquisition.
        delay_after_preparation
            Time between preparation pulse and start of data acquisition. During this time, standard longitudinal
            relaxation occurs.

        """
        super().__init__()
        sampling_time = torch.as_tensor(sampling_time)
        self.sampling_time = torch.nn.Parameter(sampling_time, requires_grad=sampling_time.requires_grad)
        repetition_time = torch.as_tensor(repetition_time)
        self.repetition_time = torch.nn.Parameter(repetition_time, requires_grad=repetition_time.requires_grad)
        m0_scaling_preparation = torch.as_tensor(m0_scaling_preparation)
        self.m0_scaling_preparation = torch.nn.Parameter(
            m0_scaling_preparation, requires_grad=m0_scaling_preparation.requires_grad
        )
        delay_after_preparation = torch.as_tensor(delay_after_preparation)
        self.delay_after_preparation = torch.nn.Parameter(
            delay_after_preparation, requires_grad=delay_after_preparation.requires_grad
        )

    def forward(self, m0: torch.Tensor, t1: torch.Tensor, flip_angle: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply transient steady state signal model.

        Parameters
        ----------
        m0
            equilibrium signal / proton density
            with shape (... other, coils, z, y, x)
        t1
            longitudinal relaxation time T1
            with shape (... other, coils, z, y, x)
        flip_angle
            flip angle of data acquisition in rad
            with shape (... other, coils, z, y, x)

        Returns
        -------
            signal
            with shape (time ... other, coils, z, y, x)
        """
        m0_ndim = m0.ndim

        # -1 for time
        sampling_time = self.expand_tensor_dim(self.sampling_time, m0_ndim - (self.sampling_time.ndim - 1))

        repetition_time = self.expand_tensor_dim(self.repetition_time, m0_ndim - self.repetition_time.ndim)
        m0_scaling_preparation = self.expand_tensor_dim(
            self.m0_scaling_preparation, m0_ndim - self.m0_scaling_preparation.ndim
        )
        delay_after_preparation = self.expand_tensor_dim(
            self.delay_after_preparation, m0_ndim - self.delay_after_preparation.ndim
        )

        # effect of preparation pulse
        m_start = m0 * m0_scaling_preparation

        # relaxation towards M0
        m_start = m0 + (m_start - m0) * torch.exp(-(delay_after_preparation / t1))

        # transient steady state
        ln_cos_tr = torch.log(torch.cos(flip_angle)) / repetition_time
        r1_star = 1 / t1 - ln_cos_tr
        m0_star = m0 / (1 - t1 * ln_cos_tr)
        signal = m0_star + (m_start - m0_star) * torch.exp(-sampling_time * r1_star)
        return (signal,)
