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

from mrpro  import __version__ as project_version
config = SphinxConfig("../../pyproject.toml", globalns=globals(), config_overrides = {"version": project_version})
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
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_nb',
    'sphinx.ext.mathjax',
    'sphinx-mathjax-offline'
]
autosummary_generate = True
autosummary_imported_members = False
autosummary_ignore_module_all = False
autodoc_member_order = 'groupwise'
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {'.rst': 'restructuredtext', '.txt': 'restructuredtext', '.md': 'markdown'}

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]
nb_execution_mode = "off"

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
    'pygment_light_style': 'default',
    'pygment_dark_style': 'github-dark',
    'show_toc_level': 3,
    'icon_links': [
        {
            # Label for this link
            'name': 'GitHub',
            # URL where the link will redirect
            'url': 'https://github.com/PTB-MR/mrpro',
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            'icon': 'fa-brands fa-github',
        },
    ],
}

def setup(app):
    # forces mathjax on all pages
    app.set_html_assets_policy('always')
