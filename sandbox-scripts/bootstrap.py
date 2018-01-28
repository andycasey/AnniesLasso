
""" Bootstrap the training set to explore label uncertainties. """

import numpy as np
import os
from six.moves import cPickle as pickle
from astropy.table import Table

import AnniesLasso as tc



bootstrap_samples = 32
scale_factor, Lambda = (2.0, 10**3)



# Data.
PATH, CATALOG, FILE_FORMAT = ("", "apogee-rg.fits",
    "apogee-rg-custom-normalization-{}.memmap")

# Load the data.
labelled_set = Table.read(os.path.join(PATH, CATALOG))
dispersion = np.memmap(os.path.join(PATH, FILE_FORMAT).format("dispersion"),
    mode="c", dtype=float)
normalized_flux = np.memmap(
    os.path.join(PATH, FILE_FORMAT).format("flux"),
    mode="c", dtype=float).reshape((len(labelled_set), -1))
normalized_ivar = np.memmap(
    os.path.join(PATH, FILE_FORMAT).format("ivar"),
    mode="c", dtype=float).reshape(normalized_flux.shape)

# TODO:
elements = [label_name for label_name in labelled_set.dtype.names \
    if label_name not in ("PARAM_M_H", "SRC_H") and label_name.endswith("_H")]

# Identify the training set used previously.
np.random.seed(123) # For reproducibility.
training_set = (np.random.randint(0, 10, len(labelled_set)) > 0)

# For the purposes of ensuring that no validation stars get in here, just ignore
# them entirely:
N_training_set = sum(training_set)
labelled_set = labelled_set[training_set]
normalized_flux = normalized_flux[training_set]
normalized_ivar = normalized_ivar[training_set]

for i in range(bootstrap_samples):

    resample_indices = np.random.randint(0, N_training_set, size=N_training_set)

    # Create a spectral model.
    model = tc.L1RegularizedCannonModel(labelled_set[resample_indices],
        normalized_flux[resample_indices], normalized_ivar[resample_indices],
        dispersion=dispersion, threads=8)

    model.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(
        labelled_set, tc.vectorizer.polynomial.terminator(elements, 2),
        scale_factor=scale_factor)

    model.s2 = 0.0
    model.regularization = Lambda

    model.train()

    model._set_s2_by_hogg_heuristic()

    model.save("bootstrap-resampled-model-{}.pkl".format(i), overwrite=True)
