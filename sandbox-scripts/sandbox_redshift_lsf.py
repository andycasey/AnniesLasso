
""" Sandbox area for testing redshift and LSF fitting at test time. """

import numpy as np
import os
from six.moves import cPickle as pickle
from astropy.table import Table

import AnniesLasso as tc

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

# Identify the training set used previously.
np.random.seed(123) # For reproducibility.
training_set = (np.random.randint(0, 10, len(labelled_set)) > 0)

if not os.path.exists("testing-redshift-model.pkl"):

    # Create a spectral model.
    model = tc.L1RegularizedCannonModel(labelled_set[training_set],
        normalized_flux[training_set], normalized_ivar[training_set],
        dispersion=dispersion, threads=8)

    model.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(labelled_set,
        tc.vectorizer.polynomial.terminator(("TEFF", "LOGG", "FE_H"), 2))

    model.s2 = 0.0
    model.regularization = 0

    model.train()
    model._set_s2_by_hogg_heuristic()

    model.save("testing-redshift-model.pkl", True)


else:
    model = tc.load_model("testing-redshift-model.pkl")


foo = model.fit(normalized_flux[650], normalized_ivar[650],  model_redshift=True,
    full_output=True)
bar = model.fit(normalized_flux[650], normalized_ivar[650], full_output=True)

moo = model.fit(normalized_flux[650], normalized_ivar[650],  model_lsf=True,
    full_output=True)
cow = model.fit(normalized_flux[650], normalized_ivar[650], model_lsf=True,
    model_redshift=True, full_output=True)
