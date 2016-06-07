
"""
Setup the models for the grid search for Lambda and f.
"""

import numpy as np
import os
from astropy.table import Table
from six.moves import cPickle as pickle

import AnniesLasso as tc

np.random.seed(123) # For reproducibility.

# Data.
PATH, CATALOG, FILE_FORMAT = ("", "apogee-rg.fits",
    "apogee-rg-custom-normalization-{}.memmap")

# Load the data.
labelled_set = Table.read(os.path.join(PATH, CATALOG))
dispersion = np.memmap(os.path.join(PATH, FILE_FORMAT).format("dispersion"),
    mode="r", dtype=float)
normalized_flux = np.memmap(
    os.path.join(PATH, FILE_FORMAT).format("flux"),
    mode="c", dtype=float).reshape((len(labelled_set), -1))
normalized_ivar = np.memmap(
    os.path.join(PATH, FILE_FORMAT).format("ivar"),
    mode="c", dtype=float).reshape(normalized_flux.shape)

elements = [label_name for label_name in labelled_set.dtype.names \
    if label_name not in ("PARAM_M_H", "SRC_H") and label_name.endswith("_H")]

# Split up the data into ten random subsets.
q = np.random.randint(0, 10, len(labelled_set)) % 10

validate_set = (q == 0)
train_set = (~validate_set)


# Save the validate flux and ivar to disk.
train_flux = np.memmap(os.path.join(PATH, FILE_FORMAT).format("flux-train"),
    mode="w+", dtype=float, shape=normalized_flux[train_set].shape)
train_flux[:] = normalized_flux[train_set]
train_flux.flush()
del train_flux

train_ivar = np.memmap(os.path.join(PATH, FILE_FORMAT).format("ivar-train"),
    mode="w+", dtype=float, shape=normalized_ivar[train_set].shape)
train_ivar[:] = normalized_ivar[train_set]
train_ivar.flush()
del train_ivar



###
scale_factors = [0.5, 1, 2, 5, 10, 20, 30, 40, 50]
Lambdas = 10**np.array([3, 3.5, 4, 4.5, 5])

for scale_factor in scale_factors:
    for Lambda in Lambdas:

        model = tc.L1RegularizedCannonModel(labelled_set[train_set],
            normalized_flux[train_set], normalized_ivar[train_set],
            dispersion=dispersion)

        model.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(
            labelled_set,
            tc.vectorizer.polynomial.terminator(["TEFF", "LOGG", "FE_H"], 2),
            scale_factor=scale_factor)

        model.s2 = 0.0
        model.regularization = Lambda
        model._labelled_set = model._labelled_set.as_array()
        model._normalized_flux = os.path.join(PATH, FILE_FORMAT).format("flux-train")
        model._normalized_ivar = os.path.join(PATH, FILE_FORMAT).format("ivar-train")

        # Save the model.
        model.save("gridsearch-{0:.1f}-{1:.1f}.model".format(
            scale_factor, np.log10(Lambda)), include_training_data=True)

# Next: Train all the models.

# Calculate sparsity for all models --> save this.

# Do validation for all models --> save this and chi-sq values.


