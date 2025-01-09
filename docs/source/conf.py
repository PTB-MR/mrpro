# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import ast
import dataclasses
import inspect
import os
import shutil
import sys
from pathlib import Path

from sphinx_pyproject import SphinxConfig

from sphinx_pyproject import SphinxConfig
from sphinx.ext.autodoc import ClassDocumenter, MethodDocumenter, AttributeDocumenter, PropertyDocumenter
from sphinx.util.inspect import isstaticmethod, isclassmethod

from mrpro import __version__ as project_version

config = SphinxConfig('../../pyproject.toml', globalns=globals(), config_overrides={'version': project_version})
sys.path.insert(0, os.path.abspath('../../src'))  # Source code dir relative to this file


project = name
copyright = '2023, Physikalisch-Technische Bundesanstalt (PTB) Berlin'
author = author
version = version

extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_github_style',
    'sphinx.ext.napoleon',
    'myst_nb',
    'sphinx.ext.mathjax',
    'sphinx-mathjax-offline',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
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
napoleon_use_rtype = False
typehints_defaults = 'comma'
typehints_use_signature = True
typehints_use_signature_return = True
typehints_use_rtype = False
typehints_document_rtype = False
autosummary_generate = True
autosummary_imported_members = False
autosummary_ignore_module_all = False
autodoc_member_order = 'groupwise'
autodoc_preserve_defaults = True
autodoc_class_signature = 'separated'
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {'.rst': 'restructuredtext', '.txt': 'restructuredtext', '.md': 'markdown'}
myst_enable_extensions = [
    'amsmath',
    'dollarmath',
]
nb_execution_mode = 'off'
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
    'github_url' : 'https://github.com/PTB-MR/mrpro/main'

}
linkcode_blob = html_context['github_version']


def get_lambda_source(obj):
    """Convert lambda to source code."""
    source = inspect.getsource(obj)
    for node in ast.walk(ast.parse(source.strip())):
        if isinstance(node, ast.Lambda):
            return ast.unparse(node.body)

class DefaultValue:
    """Used to store default values of dataclass fields with default factory."""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        """This is called by sphinx when rendering the default value."""
        return self.value


def rewrite_dataclass_init_default_factories(app, obj, bound_method) -> None:
    """Replace default fields in dataclass.__init__."""
    if (
        not 'init' in str(obj)
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
            if field.default_factory.__name__ == '<lambda>':
                defaults[field.name] = DefaultValue(get_lambda_source(field.default_factory))
            else:
                defaults[field.name] = DefaultValue(field.default_factory.__name__ + '()')
    new_defaults = tuple(defaults.get(name, param.default) for name, param in parameters.items() if param.default != inspect._empty)
    obj.__defaults__ = new_defaults


class CustomClassDocumenter(ClassDocumenter):
    """
    Custom Documenter to reorder class members
    """

    def sort_members(
        self, documenters: list[tuple['Documenter', bool]], order: str
    ) -> list[tuple['Documenter', bool]]:
        """
        Sort the given member list with custom logic for `groupwise` ordering.
        """
        if order == "groupwise":
            if not self.parse_name() or not self.import_object():
                return documenters
            # Split members into groups (non-inherited,inherited)
            static_methods = [],[]
            class_methods = [],[]
            special_methods = [],[]
            instance_methods = [],[]
            attributes = [],[]
            properties = [],[]
            other=[],[]
            others_methods = []
            init_method = []


            for documenter in documenters:
                doc = documenter[0]
                parsed = doc.parse_name() and doc.import_object()
                inherited = parsed and doc.object_name not in self.object.__dict__
                if isinstance(doc, AttributeDocumenter):
                    attributes[inherited].append(documenter)
                elif isinstance(doc, PropertyDocumenter):
                    properties[inherited].append(documenter)
                elif isinstance(doc,MethodDocumenter):
                    if not parsed:
                        others_methods.append(documenter)
                        continue
                    if doc.object_name == "__init__":
                        init_method.append(documenter)
                    elif dataclasses.is_dataclass(self.object) and doc.object_name=="__new__":
                        ...
                    elif doc.object_name[:2]=="__":
                        special_methods[inherited].append(documenter)
                    elif isclassmethod(doc.object):
                        class_methods[inherited].append(documenter)
                    elif isstaticmethod(doc.object):
                        static_methods[inherited].append(documenter)
                    else:
                        instance_methods[inherited].append(documenter)
                else:
                    other[inherited].append(documenter)
                    continue
            # Combine groups in the desired order
            constructors = init_method + class_methods[0] + class_methods[1]
            methods = instance_methods[0] + instance_methods[1] + others_methods + static_methods[0] + static_methods[1] + special_methods[0] + special_methods[1]
            return constructors+ attributes[0]+attributes[1] + properties[0]+properties[1]+methods + other[0]+other[1]
        else:
            return super().sort_members(documenters, order)




def sync_notebooks(source_folder, dest_folder):
    """
    Synchronize files from the source to the destination folder, copying only new or updated files.
    """
    dest = Path(dest_folder)
    dest.mkdir(parents=True, exist_ok=True)
    for src_file in Path(source_folder).iterdir():
        if src_file.is_file():
            dest_file = dest / src_file.name
            if not dest_file.exists() or src_file.stat().st_mtime > dest_file.stat().st_mtime:
                shutil.copy2(src_file, dest_file)

def setup(app):
    app.set_html_assets_policy('always') # forces mathjax on all pages
    app.connect('autodoc-before-process-signature', rewrite_dataclass_init_default_factories)
    app.add_autodocumenter(CustomClassDocumenter)
    sync_notebooks(app.srcdir.parent.parent/'examples'/'notebooks', app.srcdir/'_notebooks')

