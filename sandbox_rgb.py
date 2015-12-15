
"""
Perform the 17-label training for a few Lambda parameters for the red giant
branch sample.
"""

import numpy as np
import os
from sys import maxsize
from astropy.table import Table

import AnniesLasso as tc

np.random.seed(123) # For reproducibility.

# Some configurable options..
threads = 10

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
q = np.random.randint(0, maxsize, len(labelled_set)) % 10

validate_set = (q == 0)
test_set = (q == 9)
train_set = (~validate_set) * (~test_set)
assert np.sum(np.hstack([validate_set, test_set, train_set])) == q.size

# Create a vectorizer for all models.
vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(labelled_set,
    tc.vectorizer.polynomial.terminator(["TEFF", "LOGG"] + elements, 2))

# Create a zero regularization model and train it on 7/10ths the subset.
standard_cannon = tc.CannonModel(labelled_set[train_set],
    normalized_flux[train_set], normalized_ivar[train_set],
    dispersion=dispersion, threads=threads)
standard_cannon.vectorizer = vectorizer
standard_cannon.train()

raise a

standard_cannon.save("apogee-rg-standard-cannon.model", overwrite=True)

# Predict labels for the last 1/10th (test) set and compare them to ASCAP.
aspcap_labels = np.array([labelled_set[label_name][test_set] \
    for label_name in vectorizer.label_names]).T
standard_cannon_predicted_labels = standard_cannon.fit(
    normalized_flux[test_set], normalized_ivar[test_set])


# Create a regularized Cannon model to try at different Lambda values.
regularized_cannon = tc.L1RegularizedCannonModel(labelled_set[train_set],
    normalized_flux[train_set], normalized_ivar[train_set],
    dispersion=dispersion, threads=threads)
regularized_cannon.vectorizer = vectorizer


# For ~50 pixels, try the following Lambda values.




raise OKNowWhat


# Choose a single Lambda value for all pixels.


# Create a 10^5 regularization model and train it on 7/10ths the subset.
regularized_cannon.train()

regularized_cannon.save("apogee-rg-regularized-cannon.model", overwrite=True)

# Predict labels for the last 1/10th set and compare them to ASPCAP.
regularized_cannon_predicted_labels = regularized_cannon.fit(
    normalized_flux[test_set], normalized_ivar[test_set])


# Are we doing as good, or better than ASPCAP?


# How many terms did we remove? --> How has the sparsity changed?

