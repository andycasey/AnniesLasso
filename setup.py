#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from setuptools import setup, find_packages
from codecs import open
from os import path, system
from re import compile as re_compile

# For convenience.
if sys.argv[-1] == "publish":
    system("python setup.py sdist upload")
    sys.exit()

def read(filename):
    kwds = {"encoding": "utf-8"} if sys.version_info[0] >= 3 else {}
    with open(filename, **kwds) as fp:
        contents = fp.read()
    return contents

# Get the version information.
here = path.abspath(path.dirname(__file__))
vre = re_compile("__version__ = \"(.*?)\"")
version = vre.findall(read(path.join(here, "AnniesLasso", "__init__.py")))[0]

setup(
    name="AnniesLasso",
    version=version,
    author="Andrew R. Casey, David W. Hogg, Melissa K. Ness",
    author_email="arc@ast.cam.ac.uk",
    description="The Cannon 2: Compressed sensing edition",
    long_description=read(path.join(here, "README.md")),
    url="http://thecannon.io",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    keywords="The Cannon Annie's Lasso",
    packages=find_packages(exclude=["documents", "tests"]),
    install_requires=["numpy", "scipy", "six", "sharedmem"],
    extras_require={
        "test": ["coverage"]
    },
    package_data={
        "": ["LICENSE"],
    },
    include_package_data=True,
    data_files=None,
    entry_points={
        "console_scripts": [
            "tc = AnniesLasso.__main__:main",
            "cannon = AnniesLasso.__main__:main",
            "cannon-local = AnniesLasso.local_cannon:main"
        ]
    }
)
