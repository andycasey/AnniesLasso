# The Cannon 2: The Compressed Sensing Edition

If we take *The Cannon* to large numbers of labels (say chemical abundances),
the model complexity grows very fast.  At the same time, we know that most 
chemicals affect very few wavelengths in the spectrum; that is, we know that the 
problem is sparse.  Here we try to use standard methods to discover and enforce
 sparsity.

[![Build Status](https://img.shields.io/travis/andycasey/AnniesLasso/master.svg)](https://travis-ci.org/andycasey/AnniesLasso)
[![Coverage Status](https://img.shields.io/coveralls/andycasey/AnniesLasso/master.svg)](https://coveralls.io/github/andycasey/AnniesLasso?branch=master)
[![Scrutinizer](https://img.shields.io/scrutinizer/g/andycasey/AnniesLasso.svg?b=master)](https://scrutinizer-ci.com/g/andycasey/AnniesLasso/?branch=master)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/andycasey/AnniesLasso/blob/master/LICENSE)


## Authors
- **Andy Casey** (Cambridge)
- **David W. Hogg** (NYU) (MPIA) (SCDA)
- **Melissa K. Ness** (MPIA)
- **Hans-Walter Rix** (MPIA)
- **Anna Y. Q. Ho** (Caltech)
- **Gerry Gilmore** (Cambridge)


## License
**Copyright 2016 the authors**.
The code in this repository is released under the open-source **MIT License**.
See the file `LICENSE` for more details.


## Installation

To install:

``
pip install https://github.com/andycasey/AnniesLasso/archive/master.zip
``


## Getting Started

Let us assume that you have rest-frame continuum-normalized spectra for a set of
stars for which the stellar parameters and chemical abundances (which we will
collectively call *labels*) are known with high fidelity.  The labels for those
stars (and the locations of the spectrum fluxes and inverse variances) are
assumed to be stored in a table.  In this example all stars are assumed to be 
sampled on the same wavelength (dispersion) scale.


Here we will create and train a 3-label (effective temperature, surface gravity,
metallicity) quadratic (e.g., `Teff^2`) model:


````python
import numpy as np
from astropy.table import Table

import AnniesLasso as tc

# Load the table containing the training set labels, and the spectra.
training_set = Table.read("training_set_labels.fits")

# Here we will assume that the flux and inverse variance arrays are stored in
# different ASCII files. The end goal is just to produce flux and inverse
# variance arrays of shape (N_stars, N_pixels).
normalized_flux = np.array([np.loadtxt(star["flux_filename"]) for star in training_set])
normalized_ivar = np.array([np.loadtxt(star["ivar_filename"]) for star in training_set])

# Providing the dispersion to the model is optional, but handy later on.
dispersion = np.loadtxt("common_wavelengths.txt")

# Create the model that will run in parallel using all available cores.
model = tc.CannonModel(training_set, normalized_flux, normalized_ivar,
    dispersion=dispersion, threads=-1)

# Specify the complexity of the model:
model.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(labelled_set,
    tc.vectorizer.polynomial.terminator(("TEFF", "LOGG", "FEH"), 2))

# Train the model!
model.train()
````

You can follow this example further in the complete [Getting Started](#) tutorial.
