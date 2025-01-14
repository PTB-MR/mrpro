.. MRpro documentation

.. image:: _static/logo.svg
   :align: center
   :width: 300

.. |colab-badge| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/PTB-MR/mrpro

Welcome to MRpro's documentation!
=================================

MR image reconstruction and processing for PyTorch

| **Source code:** `<https://github.com/PTB-MR/mrpro>`_
| **Bug reports:** `<https://github.com/PTB-MR/mrpro/issues>`_
| **Try it out:** |colab-badge|
| **See our examples:** :doc:`examples`

Main Features
-------------
- **Standard file formats**
  MRpro supports the ISMRMRD format for MR raw data and DICOM for image data

- **PyTorch integration**
  All data containers utilize PyTorch tensors to ensure easy integration with PyTorch-based network schemes.

- **Cartesian and non-Cartesian trajectories**
  MRpro can reconstruct data obtained with Cartesian and non-Cartesian sampling schemes (e.g., radial, spiral). It automatically detects whether FFT or nuFFT is required to reconstruct the k-space data.

- **Pulseq support**
  If the data acquisition was carried out using a pulseq-based sequence, the seq-file can be provided to MRpro, which will automatically calculate the used trajectory.

- **Signal models**
  A range of MR signal models is implemented (e.g., T1 recovery, WASABI).

- **Regularized image reconstruction**
  Regularized image reconstruction algorithms, including wavelet-based compressed sensing and total variation regularized image reconstruction, are available.


Content
=======
.. toctree::
   :maxdepth: 2
   user_guide
   examples
   contributor_guide
   api
   faq


