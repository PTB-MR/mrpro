# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
from sphinx_pyproject import SphinxConfig
from urllib.parse import quote

from mrpro  import __version__ as project_version
config = SphinxConfig('../../pyproject.toml', globalns=globals(), config_overrides = {'version': project_version})
sys.path.insert(0, os.path.abspath('../../src'))  # Source code dir relative to this file

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = name
copyright = '2023, Physikalisch-Technische Bundesanstalt (PTB) Berlin'
author = author
version = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
  #  'sphinx.ext.linkcode',
  'sphinx_github_style',
    'sphinx.ext.napoleon',
    'myst_nb',
    'sphinx.ext.mathjax',
    'sphinx-mathjax-offline',
    'sphinx.ext.intersphinx',
  #  'sphinx_autodoc_typehints',

]
intersphinx_mapping = {
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'ismrmrd': ('https://ismrmrd.readthedocs.io/en/latest/', None),
    'einops': ('https://einops.rocks/', None),
    'python': ('https://docs.python.org/3', None),
    'pydicom': ('https://pydicom.github.io/pydicom/stable/', None),
    'pypulseq': ('https://pypulseq.readthedocs.io/en/master/', None),
    'torchkbnufft': ('https://torchkbnufft.readthedocs.io/en/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'ptwt': ('https://pytorch-wavelet-toolbox.readthedocs.io/en/latest/', None),
    'typing-extensions': ('https://typing-extensions.readthedocs.io/en/latest/', None),
}

napoleon_use_param = True
typehints_defaults = 'comma'
typehints_use_signature = True
typehints_use_signature_return = True
autosummary_generate = True
autosummary_imported_members = False
autosummary_ignore_module_all = False
autodoc_member_order = 'groupwise'
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {'.rst': 'restructuredtext', '.txt': 'restructuredtext', '.md': 'markdown'}
autodoc_class_signature = 'separated'
myst_enable_extensions = [
    'amsmath',
    'dollarmath',
]
nb_execution_mode = 'off'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_title = name
html_show_sphinx = False
html_static_path = ['_static']
html_css_files = ['custom.css']
html_logo = '_static/logo_white.svg'
html_sidebars = {'**': ['search-field', 'sidebar-nav-bs']}
html_theme_options = {
    'logo_only': True,
    'collapse_navigation': False,
}
html_context = {
    'display_github': True,
    'github_user': 'PTB-MR',
    'github_repo': 'mrpro',
    'github_version': 'main',
}
linkcode_blob = html_context['github_version']
autodoc_preserve_defaults=True


import inspect, ast,dataclasses
def get_lambda_source(obj):
    source = inspect.getsource(obj)
    for node in ast.walk(ast.parse(source.strip())):
        if isinstance(node, ast.Lambda):
            return ast.unparse(node.body)
class DefaultValue():
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return self.value

def rewrite_dataclass_init_default_factories(app, obj, bound_method) -> None:
    if (not "init" in str(obj)
        or not getattr(obj, '__defaults__', None)
        or not any(isinstance(d, dataclasses._HAS_DEFAULT_FACTORY_CLASS) for d in obj.__defaults__)
    ):
        # not an dataclass.__init__ method with default factory
        return
    parameters = inspect.signature(obj).parameters
    module = sys.modules[obj.__module__]
    class_ref = getattr(module, obj.__qualname__.split('.')[0])
    defaults = {}
    for field in dataclasses.fields(class_ref):
        if field.default_factory is not dataclasses.MISSING:
            if not field.name in parameters:
                continue
            if field.default_factory.__name__ == "<lambda>":
                defaults[field.name]  = DefaultValue(get_lambda_source(field.default_factory))
            else:
                defaults[field.name] = DefaultValue(field.default_factory.__name__ + "()")
    new_defaults = tuple(defaults.get(name,param.default) for name, param in parameters.items() if param.default!=inspect._empty)
    obj.__defaults__ = new_defaults

def setup(app):
    # forces mathjax on all pages
    app.set_html_assets_policy('always')
    # rewrite dataclass init signature
    app.connect("autodoc-before-process-signature", rewrite_dataclass_init_default_factories)

