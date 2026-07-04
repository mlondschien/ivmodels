# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "ivmodels"
copyright = "2023, Malte Londschien"
author = "Malte Londschien"

extensions = [
    "sphinx.ext.napoleon",
    "sphinxcontrib.apidoc",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "sphinx_rtd_theme",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinxcontrib.bibtex",
]


bibtex_bibfiles = ["bib.bib"]
bibtex_reference_style = "author_year"

# Each docstring contains its own `.. bibliography:: :filter: False` block and each
# module is rendered both in api.rst and in the apidoc-generated api/ivmodels.*.rst,
# so the same citation keys appear multiple times.
suppress_warnings = ["bibtex.duplicate_citation", "bibtex.duplicate_local_citation"]

# intersphinx configuration
intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
apidoc_module_dir = "../ivmodels"
apidoc_output_dir = "api"
apidoc_extra_args = ["--implicit-namespaces"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
