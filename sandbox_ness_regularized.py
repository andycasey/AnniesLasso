#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run The Cannon using a L1-regularized edition with the Ness et al. (2015)
training set.
"""

import numpy as np
from astropy.table import Table
from glob import glob

import AnniesLasso as tc

"""
Load the data files:
 - Ness_2015_clusters_dispersion.memmap
 - Ness_2015_clusters_normalized_flux.memmap
 - Ness_2015_clusters_normalized_ivar.memmap
 - Ness_2015_labelled_set.fits
"""
N = None
labelled_set = Table.read("Ness_2015_labelled_set.txt", format="ascii")
valid = labelled_set["LOGG"] > 0.1

dispersion, normalized_flux, normalized_ivar = \
    (np.memmap("Ness_2015_clusters_{}.memmap".format(_), mode="c", dtype=float)\
        for _ in ("dispersion", "normalized_flux", "normalized_ivar"))
normalized_flux = normalized_flux.reshape((len(labelled_set), -1))
normalized_ivar = normalized_ivar.reshape((len(labelled_set), -1))

labelled_set = labelled_set[valid]
normalized_flux = normalized_flux[valid]
normalized_ivar = normalized_ivar[valid]

# Specify the model.
model = tc.L1RegularizedCannonModel(labelled_set, normalized_flux[:, :N],
    normalized_ivar[:, :N], dispersion=dispersion[:N], threads=10)

model.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(labelled_set,
    tc.vectorizer.polynomial.terminator(["TEFF", "LOGG", "PARAM_M_H"], 2))

model.regularization = 0

raise a
"""
Lambdas, chi_sq, log_det, models = model.validate_regularization(
    fixed_scatter=0, Lambdas=10**np.arange(2, 8.5, 0.5),
    model_filename_format="Ness_2015_L1_validation_{}.pkl", overwrite=True)
"""

models = \
    [model.load(filename) for filename in glob("Ness_2015_L1_validation_*.pkl")]

wavelengths = [15339.0, 16811.5, 15221.5, dispersion[7700]]
fig = tc.diagnostics.pixel_regularization_effectiveness(models,
    wavelengths=wavelengths,
    latex_labels=[r"{T_{\rm eff}}", r"\log{g}", r"{\rm [M/H]}"])

raise a
