# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cookproject'
copyright = '2024, Aude de Fornel, Camille Ishac, Romain Donné'
author = 'Aude de Fornel, Camille Ishac, Romain Donné'
release = '1.0'
year = datetime.now().year

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Extensions utilisées
extensions = [
    'sphinx.ext.autodoc',        # Générer la documentation depuis les docstrings
    'sphinx.ext.napoleon',       # Support Google/Numpy docstring styles
    'sphinx.ext.viewcode',       # Lien vers le code source
    'sphinx_autodoc_typehints',  # Support des annotations de type
]


templates_path = ['_templates']
exclude_patterns = []

sys.path.insert(0, os.path.abspath('../..'))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
