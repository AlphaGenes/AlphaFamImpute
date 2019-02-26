
from setuptools import setup

from setuptools import Extension, find_packages
import numpy
import os
import glob
from sys import platform
import sys
import sysconfig
py_modules = ['AlphaFamImpute']

# py_modules += [os.path.join('src','General', 'InputOutput')]
# py_modules += [os.path.join('src','General', 'Pedigree')]
src_modules = []
src_modules += glob.glob(os.path.join('tinyhouse', '*.py'))
src_modules += glob.glob(os.path.join('FamilyImputation', '*.py'))

src_modules = [os.path.splitext(file)[0] for file in src_modules]
py_modules += src_modules

setup(
    name="AlphaFamImpute",
    version="0.1",
    author="Andrew Whalen",
    author_email="awhalen@roslin.ed.ac.uk",
    description="An imputation program for phasing and imputation in a full sib family.",
    long_description="An imputation program for phasing and imputation in a full sib family.",
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    py_modules = py_modules,

    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    entry_points = {
    'console_scripts': [
        'AlphaFamImpute=AlphaFamImpute:main'
        ],
    },
    install_requires=[
        'numpy',
        'numba'
    ]
)
