======================
Guide for Contributors
======================

Repo structure
==============
This repository uses a *pyproject.toml* file to specify all the requirements.
If you need a "normal" *requirements.txt* file, please have a look in *binder*. There you find a *requirements.txt*
automatically created from *pyproject.toml* using GitHub actions.

**.github/workflows**
    Definitions of GitHub action workflows to carry out formatting checks, run tests and automatically create this
    documentation.

**.vscode**
    Configuration files for VS Code.

**.docs**
    Files to create this documentation.

**binder**
    *environment.yml* to install MRpro when starting a binder session.

**examples**
    Python scripts showcasing how MRpro can be used. Any data needed has to be available from
    an online repository (e.g. zenodo) such that it can be automatically downloaded. The scripts 
    are automatically translated to jupyter notebooks using GitHub
    actions. Individual cells should be indicated with ``# %%``. For markdown cells use ``# %% [markdown]``.
    The translation from python script to jupyter notebook is done using
    `jupytext <https://jupytext.readthedocs.io/en/latest/>`_ . See their documentation for more details.

    After translating the scripts to notebooks, the notebooks are run and their output is converted to html and added
    to this documentation in the *Examples* section.

    We are not using notebooks directly because if contributors forget to clear all cells prior to committing then the
    content of the notebook is also version controlled with git which makes things very messy.

**mrpro/src**
    Main code for this package

**tests**
    Tests which are automatically run by pytest.
    The subfolder structure should follow the same structure as in *mrpro/src*.


src/mrpro structure
===================
**algorithms**
    Everything which does something with the data, e.g. prewhiten k-space or remove oversampling.

**data**
    All the data classes such as ``KData``, ``ImageData`` or ``CsmData``.
    As the name suggestions these should mainly contain data and meta information.
    Any functionality beyond what is absolutely required for the classes should be put as separate functions.

**operators**
    Linear and non-linear algorithms describing e.g. the transformation from image to k-space (``FourierOp``), the
    effect of receiver coils (``SensitivityOp``) or MR signal models.

**phantoms**
    Numerical phantoms useful to evaluate reconstruction algorithms.

**utils**
    Utilities such as spatial filters and also more basic functionality such as applying functions serially along the
    batch dimension (``smap``).


Naming convention
=================
We try to follow the [pep8](https://peps.python.org/pep-0008/) naming convention (e.g., all lowercase variable names,
CapWords class names). We deviate for the names of source code file names containing a single class.
These are named as the class.

We try to use descriptive variable names when applicable (e.g., ``result`` instead of ``res``, ``tolerance_squared`` instead
of ``sqtol``, ``batchsize`` instead of ``m``).

A name starting with ``n_`` is used for variables describing a number of... (e.g., ``n_coils`` instead of ``ncoils`` or
``num_coils``), variable names ending with ``_op`` for operators (e.g., ``fourier_op``). We use ``img`` as a variable name
for images.


