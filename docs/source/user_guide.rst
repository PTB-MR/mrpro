=====================
Guide for MRPro Users
=====================

MRpro is a MR image reconstruction and processing framework specifically developed to work well with pytorch.
The data classes utilize torch tensors for storing data such as MR raw data or reconstructed image data.
Where possible batch parallelisation of pytorch is utilized to speed up image reconstruction.

MRpro is designed to work directly from MR raw data using the  `MRD <https://ismrmrd.readthedocs.io/en/latest/>`_ data format.

A basic pipeline would contain the following steps:

* Reading in raw data
* Preparation for reconstruction
* Data reconstruction
* Image processing

The following provides some basic information about these steps.
For more detailed information please have a look at the notebooks in the *examples* folder.
You can easily start a binder session via the badge in the *README* and give the notebooks a try without having to
install anything.

Reading in raw data
===================
Reading in raw data from a MRD file works by creating a ``KData`` object and using the class method ``from_file``.
``KData`` contains the raw k-space data, the header information obtained from the MRD file and the k-space trajectory.
To ensure the trajectory is calculated correctly, a ``KTrajectoryCalculator`` needs to be provided.
The trajectory can either be calculated based on MRpro functionality (e.g. for a 2D radial sampling scheme), read out
from MRD or calculated from a `pulseq <http://pulseq.github.io/>`_ file.

Preparation for reconstruction
==============================
MRpro provides a range of functionality to prepare the data for image reconstruction such as:

* Noise prewhiting
* Removal of oversampling along readout direction
* Calculation of the density compensation function
* Estimation of coil sensitiviy maps
* ...

Data reconstruction
===================
As a first step for the reconstruction an acquisition model consisting of linear and non-linear operators needs to
be created. A simply acquisition model could consist of a ``SensitivityOp`` describing the effect of different
receiver coils and ``FourierOp`` describing the transform from image space to k-space taking the sampling scheme
(trajectory) into account. Additional operators describing transformations due to physiological motion or
MR signal models can be added.

Based on the acquisition model a suitable minimization function and reconstruction algorithm needs to be selected.

Depending on the choices made above the reconstruction algorithms provides images (``ImageData``) or quantitative
parametric maps (``QData``).

Image processing
================
Further processing of the reconstructed data such as quantitative parameter estimation is available.
The *examples* folder also contains notebooks which show how to carry out motion estimation from reconstructed dynamic
images.
