# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import shutil
import sys
from pathlib import Path

import nbformat
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
default_role = 'py:obj'
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {'.rst': 'restructuredtext', '.txt': 'restructuredtext', '.md': 'markdown'}

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]
nb_execution_mode = "off"
nb_merge_streams = True

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


def sync_notebooks(source_folder, dest_folder):
    """Sync notebooks from source to destination folder.

    Copy only new or updated files.
    Set execution mode to 'force' for all copied files and 'off' for all existing files.
    """
    dest = Path(dest_folder)
    dest.mkdir(parents=True, exist_ok=True)
    for src_file in Path(source_folder).iterdir():
        if src_file.is_file():
            dest_file = dest / src_file.name
            if not dest_file.exists() or src_file.stat().st_mtime > dest_file.stat().st_mtime:
                shutil.copy2(src_file, dest_file)
                print(f'Copied {src_file} to {dest_file}. Setting execution mode to "force".')
                mode = 'force'
            else:
                print(f'Existing {dest_file}. Skipping execution.')
                mode = 'off'
            content = nbformat.read(dest_file, as_version=nbformat.NO_CONVERT)
            content.metadata['mystnb'] = {'execution_mode': mode}
            nbformat.write(content, dest_file)

def setup(app):
    # forces mathjax on all pages
    app.set_html_assets_policy('always')
    sync_notebooks(app.srcdir.parent.parent/'examples'/'notebooks', app.srcdir/'_notebooks')

