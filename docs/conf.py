import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = "Hermione"

extensions = [
    "myst_parser",
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon'
]
# html_theme = "sphinx_rtd_theme"
source_suffix = [".rst", ".md"]

master_doc = 'index'
