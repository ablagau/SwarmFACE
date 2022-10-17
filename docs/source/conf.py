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
root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
#sys.path.insert(0, os.path.abspath(os.path.join(root,'SwarmFACE','plot_save')))
sys.path.insert(0, root)

#sys.path.insert(0, os.path.abspath('/home/blagau/FACpy/SwarmFACE/'))
#print(root)
#print(os.path.abspath(os.path.join(root,'SwarmFACE')))
#print(os.path.abspath(os.path.join(root,'SwarmFACE','plot_save')))
#print(sys.path)
#import os
#import sys
#from setuptools.config import read_configuration

#root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))

#conf_dict = read_configuration(os.path.join(root,'setup.cfg'))

#for package in conf_dict['options']['packages']:
    #sys.path.insert(0, os.path.abspath(os.path.join(root,package)))


# -- Project information -----------------------------------------------------

project = 'SwarmFACE'
copyright = '2022, Adrian Blagau, Joachim Vogt'
author = 'Adrian Blagau, Joachim Vogt'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.coverage'    
    ]
source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
