
import os
import numpy as np
from astropy.table import Table


import AnniesLasso as tc



a = tc.load_model("gridsearch-2.0-3.0.model", threads=8)

# Load the data.
PATH, CATALOG, FILE_FORMAT = ("", "apogee-rg.fits",
    "apogee-rg-custom-normalization-{}.memmap")

labelled_set = Table.read(os.path.join(PATH, CATALOG))
dispersion = np.memmap(os.path.join(PATH, FILE_FORMAT).format("dispersion"),
    mode="r", dtype=float)
normalized_flux = np.memmap(
    os.path.join(PATH, FILE_FORMAT).format("flux"),
    mode="c", dtype=float).reshape((len(labelled_set), -1))
normalized_ivar = np.memmap(
    os.path.join(PATH, FILE_FORMAT).format("ivar"),
    mode="c", dtype=float).reshape(normalized_flux.shape)

# Split up the data into ten random subsets.
np.random.seed(123) # For reproducibility.
q = np.random.randint(0, 10, len(labelled_set)) % 10

validate_set = (q == 0)
train_set = (~validate_set)


a._dispersion = dispersion
a._labelled_set = labelled_set[train_set]
a._normalized_flux = normalized_flux[train_set]
a._normalized_ivar = normalized_ivar[train_set]


a._set_s2_by_hogg_heuristic()
a.train()
a.save("gridsearch-2.0-3.0-s2-heuristically-set.model", overwrite=True)

raise a

b = tc.L1RegularizedCannonModel(labelled_set[validate_set], 
    normalized_flux[validate_set], normalized_ivar[validate_set], dispersion,
    threads=4)
b.regularization = 1000.0
b.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(
    labelled_set[validate_set],
    tc.vectorizer.polynomial.terminator(["TEFF", "LOGG", "FE_H"], 2))


