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

sys.path.insert(0, os.path.abspath('../../src'))  # Source code dir relative to this file


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MRPro'
copyright = '2023, Physikalisch-Technische Bundesanstalt (PTB) Berlin'
author = 'Christoph Kolbitsch, Patrick Schuenke, Felix Zimmermann, David Schote'
release = '0.0.1'  # TODO: import this from package
version = '0.0.1'  # TODO: import this from package


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {'.rst': 'restructuredtext', '.txt': 'restructuredtext', '.md': 'markdown'}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_title = 'MRPro'
html_logo = '_static/img/logo.jpg'
html_show_sphinx = False
html_static_path = ['_static']
html_css_files = ['custom.css']
html_sidebars = {'**': ['search-field', 'sidebar-nav-bs']}
html_theme_options = {
    'logo': {'text': 'MRPro'},
    'pygment_light_style': 'default',
    'pygment_dark_style': 'github-dark',
    'show_toc_level': 3,
    'icon_links': [
        {
            # Label for this link
            'name': 'GitHub',
            # URL where the link will redirect
            'url': 'https://github.com/ckolbPTB/mrpro',
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            'icon': 'fa-brands fa-github',
        },
    ],
}
