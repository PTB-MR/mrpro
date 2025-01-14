==========
User Guide
==========

MRpro is a MR image reconstruction and processing framework specifically developed to work well with pytorch.
The data classes utilize `torch.Tensor` for storing data such as MR raw data or reconstructed image data,
operators are implemented as `torch.nn.Module`
Where possible batch parallelisation of pytorch is utilized to speed up image reconstruction.

Installation
============

MRpro is available on `pypi <https://pypi.org/project/mrpro/>`_ and can be installed with::

    pip install mrpro

To install additional dependencies used in our example notebooks, use::

    pip install mrpro[notebook]

You can also install the latest development directly from github using::

    pip install "git+https://github.com/PTB-MR/mrpro"


Usage
=====
MRpro is designed to work directly from MR raw data using the `MRD <https://ismrmrd.readthedocs.io/en/latest/>`_ data format.

A basic pipeline would contain the following steps:

* Reading in raw data
* Preparation for reconstruction
* Data reconstruction
* Image processing

.. |colab-badge| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/PTB-MR/mrpro

The following provides some basic information about these steps.
For more detailed information please have a look at the :doc:`examples`.
You can easily launch notebooks via the |colab-badge| badge and give the notebooks a try without having to
install anything.

Reading in raw data
-------------------
Reading in raw data from a MRD file works by creating a `mrpro.data.KData` object and using the class method `~mrpro.data.KData.from_file`.
`~mrpro.data.KData` contains the raw k-space data, the header information obtained from the MRD file and the k-space trajectory.
To ensure the trajectory is calculated correctly, a `~mrpro.data.traj_calculators.KTrajectoryCalculator` needs to be provided.
The trajectory can either be calculated based on MRpro functionality (e.g. for a 2D radial sampling scheme), read out
from MRD or calculated from a `pulseq <http://pulseq.github.io/>`_ file. See `~mrpro.data.traj_calculators`
for available trajectory calculators and :doc:`_notebooks/comparison_trajectory_calculators` for an example.


.. note::
    In MRpro, we use the convention ``(z, y, x)`` for spatial dimensions and ``(k2, k1, k0)`` for k-space dimensions.
    Here, `k0` is the readout direction, `k1` and `k2` are phase encoding directions.
    The full shape of a multi-slice 2D k-space data, for example, is ``(other, coil, 1, k1, k0)`` where `other` will be the different slices.
    In general, `other` can be any number of additional dimensions.

.. note::
    The trajectory is expected to be defined within the space of the `encoding_matrix`, e.g. if the
    `encoding_matrix` is defined as ``(z=1, y=256, x=256)``, then a fully sampled Cartesian trajectory without partial
    echo or partial Fourier is expected to be within ``[-128, 127]`` along both readout and phase encoding.

Preparation for reconstruction
------------------------------
MRpro provides a range of functionality to prepare the data for image reconstruction such as:

* Noise prewhiting
* Removal of oversampling along readout direction
* Calculation of the density compensation function
* Estimation of coil sensitivity maps
* Fourier transformation

Data reconstruction
-------------------
MRpro provides a flexible framework for MR image reconstruction. We provide some high level functions for commonly used
reconstruction algorithms in `mrpro.algorithms.reconstruction`, such as
`~mrpro.algorithms.reconstruction.RegularizedIterativeSENSEReconstruction`. We also provide all building blocks to
create custom reconstruction algorithms and do manual reconstructions.

As a first step for a new reconstruction, an acquisition model consisting of linear and non-linear operators can be created.
A simply acquisition model could consist of a `~mrpro.operators.SensitivityOp` describing the effect of different
receiver coils and `~mrpro.operators.FourierOp` describing the transform from image space to k-space taking the sampling scheme
(trajectory) into account. Additional operators describing transformations due to physiological motion or
MR signal models can be added. See `~mrpro.operators` for a list of available operators.
All operators take one or more tensors as input and return a tuple of one or more tensors as output.
Operators can be chained using ``@`` to form a full acquisition model. We also support addition, multiplication, etc.
between operators.

Based on the acquisition model either a suitable optimizer from `mrpro.algorithms.optimizers` can be selected
or a new optimizer using pytorch functions can be created.

See for examples  :doc:`_notebooks/cartesian_reconstruction`, :doc:`_notebooks/direct_reconstruction`, and :doc:`_notebooks/iterative_sense_reconstruction_radial2D`

Image processing
----------------
Further processing of the reconstructed data such as quantitative parameter estimation is available.
Our examples contain a notebook showing how to read in DICOM images and perform qMRI parameter estimation using
a non-linear optimizer: :doc:`_notebooks/qmri_sg_challenge_2024_t1`,


Citation
========
We are currently preparing a manuscript for MRpro. In the meantime, please cite:

Zimmermann, F. F., Schuenke, P., Brahma, S., Guastini, M., Hammacher, J., Kofler, A., Kranich Redshaw, C., Lunin, L., Martin, S., Schote, D., & Kolbitsch, C. (2024).
MRpro - PyTorch-based MR image reconstruction and processing package
`10.5281/zenodo.14509599 <https://doi.org/10.5281/zenodo.14509599>`_
