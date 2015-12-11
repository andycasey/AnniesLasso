#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run The Cannon using a L1-regularized edition with the Ness et al. (2015)
training set.
"""

import numpy as np
from astropy.table import Table

import AnniesLasso as tc

"""
The data files are:

Ness_2015_clusters_dispersion.memmap
Ness_2015_clusters_normalized_flux.memmap
Ness_2015_clusters_normalized_ivar.memmap
Ness_2015_labelled_set.fits
"""
labelled_set = Table.read("Ness_2015_labelled_set.fits")

data_filename_format = "Ness_2015_clusters_{}.memmap"
dispersion, normalized_flux, normalized_ivar = \
    (np.memmap(data_filename_format.format(descr), mode="c", dtype=float) \
        for descr in ("dispersion", "normalized_flux", "normalized_ivar"))

normalized_flux = normalized_flux.reshape((len(labelled_set), -1))
normalized_ivar = normalized_ivar.reshape((len(labelled_set), -1))


model = tc.RegularizedCannonModel(labelled_set, normalized_flux,
    normalized_ivar, dispersion=dispersion, threads=10)

model.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(labelled_set,
    tc.vectorizer.polynomial.terminator(["TEFF", "LOGG", "PARAM_M_H"], 2))

Lambdas, chi_sq, log_det, models = model.validate_regularization(
    fixed_scatter=0, Lambdas=10**np.arange(2, 8.5, 0.5),
    model_filename_format="Ness_2015_L1_validation_{}.pkl", overwrite=True)

raise a
