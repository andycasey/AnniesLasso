
import os
import numpy as np

from astropy.table import Table

import AnniesLasso as tc

PATH = "/Users/arc/research/apogee/"
CATALOG = "apogee-rc-DR12.fits"
MEMMAP_FILE_FORMAT = "apogee-rc-{}.memmap"

N = 500

# Load up the data
labelled_set = Table.read(os.path.join(PATH, CATALOG))
dispersion = np.memmap(
    os.path.join(PATH, MEMMAP_FILE_FORMAT).format("dispersion"),
    mode="c", dtype=float).flatten()
normalized_flux = np.memmap(
    os.path.join(PATH, MEMMAP_FILE_FORMAT).format("normalized-flux"),
    mode="c", dtype=float).reshape((len(labelled_set), -1))
normalized_ivar = np.memmap(
    os.path.join(PATH, MEMMAP_FILE_FORMAT).format("normalized-ivar"),
    mode="c", dtype=float).reshape(normalized_flux.shape)


# Issue: if training a complex model (or a simple one??), parallel can be slower
# than serial if memory mapped arrays are used and the mode for opening them is
# set to 'r'

model = tc.RegularizedCannonModel(
    labelled_set[:N], normalized_flux[:N], normalized_ivar[:N],
    threads=1, dispersion=dispersion)

model.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(
    model.labelled_set,
    "TEFF^2 + LOGG^2 + PARAM_M_H^2 + TEFF + LOGG + PARAM_M_H + "
    "TEFF * LOGG + TEFF * PARAM_M_H + LOGG * PARAM_M_H")

model.regularization = 0.
model.train()


# Calculate abundances for all the other stars in the sample!
#testing_set_labels = model.pre
# Plot how we are compared to APOGEE