
"""
Investigate the effect of different scale factors in the polynomial vectorizer
terms.

"""

import cPickle as pickle
import numpy as np
import os
from sys import maxsize
from astropy.table import Table

import AnniesLasso as tc

np.random.seed(123) # For reproducibility.


# Data.
PATH, CATALOG, FILE_FORMAT = ("/Users/arc/research/apogee", "apogee-rg.fits",
    "apogee-rg-{}.memmap")

# Load the data.
labelled_set = Table.read(os.path.join(PATH, CATALOG))
dispersion = np.memmap(os.path.join(PATH, FILE_FORMAT).format("dispersion"),
    mode="r", dtype=float)
normalized_flux = np.memmap(
    os.path.join(PATH, FILE_FORMAT).format("normalized-flux"),
    mode="r", dtype=float).reshape((len(labelled_set), -1))
normalized_ivar = np.memmap(
    os.path.join(PATH, FILE_FORMAT).format("normalized-ivar"),
    mode="r", dtype=float).reshape(normalized_flux.shape)

elements = [label_name for label_name in labelled_set.dtype.names \
    if label_name not in ("PARAM_M_H", "SRC_H") and label_name.endswith("_H")]

# Split up the data into ten random subsets.
mod = 10
q = np.random.randint(0, maxsize, len(labelled_set)) % mod

validate_set = (q == 0)
test_set = (q == (mod - 1))
train_set = (~validate_set) * (~test_set)
assert np.sum(np.hstack([validate_set, test_set, train_set])) == q.size




# Calculate sparsity, etc.
def calculate_sparsity(model, tolerance=1e-6):

    is_nonzero = np.abs(model.theta) > tolerance

    N = len(model.vectorizer.label_names)
    sparsity_first_order_derivatives = np.mean(is_nonzero[:, 1:1+N])
    sparsity_second_order_derivatives = np.mean(is_nonzero[:, 1+N:])
    sparsity_total = np.mean(is_nonzero[:, 1:])

    return (sparsity_first_order_derivatives, sparsity_second_order_derivatives,
        sparsity_total)


# Create two Cannon models with regularization.
regularization = 10**5.0
wavelengths = [
    16795.77313988085, # --> continuum
    15339.0, # --> Teff sensitivity
    15770.107823467057, # Al
    15684.347503529489, # Cr
]
pixels = np.searchsorted(dispersion, wavelengths)
dispersion = dispersion[pixels]

train_normalized_flux = normalized_flux[train_set]
train_normalized_flux = train_normalized_flux[:, pixels]
train_normalized_ivar = normalized_ivar[train_set]
train_normalized_ivar = train_normalized_ivar[:, pixels]

models = []
metadata = []
scale_factors = 10**np.arange(0, 5.1, 0.5)
for scale_factor in scale_factors:

    vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(labelled_set,
        tc.vectorizer.polynomial.terminator(["TEFF", "LOGG"] + elements, 2),
        scale_factor=scale_factor)


    model = tc.L1RegularizedCannonModel(labelled_set[train_set],
        train_normalized_flux, train_normalized_ivar, dispersion=dispersion)
    model.vectorizer = vectorizer
    model.regularization = regularization
    model.s2 = 0.0

    model.train(fixed_scatter=True, use_neighbouring_pixel_theta=False)

    sparsity = calculate_sparsity(model, tolerance=1e-6*scale_factor)
    print("Scale factor: {:.2e}".format(scale_factor))
    print("Sparsity: {0:.2f} {1:.2f} {2:.2f}".format(*sparsity))
    models.append(model)
    metadata.append(sparsity)


for scale_factor, sparsity in zip(scale_factors, metadata):
    print("Scale factor: {0:.2e} w/ {1:.2f} {2:.2f} {3:.2f}".format(scale_factor,
        *sparsity))

