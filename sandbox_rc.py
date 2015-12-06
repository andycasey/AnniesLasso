
import os
import numpy as np

from astropy.table import Table

import AnniesLasso

PATH = "/Users/arc/research/apogee/"
CATALOG = "apogee-rc-DR12.fits"
MEMMAP_FILE_FORMAT = "apogee-rc-{}.memmap"

model_filename = "rc-trained-abundances.pkl"

N = 2000

# Load up the data
training_labels = Table.read(os.path.join(PATH, CATALOG))
dispersion = 10**(np.memmap(
    os.path.join(PATH, MEMMAP_FILE_FORMAT).format("dispersion"),
    mode="c", dtype=float).flatten())
training_fluxes = np.memmap(
    os.path.join(PATH, MEMMAP_FILE_FORMAT).format("flux"),
    mode="c", dtype=float).reshape((len(training_labels), -1))
training_flux_uncertainties = np.memmap(
    os.path.join(PATH, MEMMAP_FILE_FORMAT).format("flux-uncertainties"),
    mode="c", dtype=float).reshape(training_fluxes.shape)


# Issue: if training a complex model (or a simple one??), parallel can be slower
# than serial if memory mapped arrays are used and the mode for opening them is
# set to 'r'

model = AnniesLasso.CannonModel(
    training_labels[:N], training_fluxes[:N], training_flux_uncertainties[:N],
    threads=4, dispersion=dispersion)

if not os.path.exists(model_filename):

    element_labels = [label for label in model.training_labels.dtype.names \
        if label.endswith("_H") and label != "PARAM_M_H"]

    label_vector = " + ".join(["TEFF^4",
        AnniesLasso.utils.build_label_vector(["TEFF", "LOGG"], 3, 2)])

    for element_label in element_labels:
        label_vector += " + {0}^2 + {0} + {0}*TEFF + {0}*LOGG".format(element_label)

    model.label_vector = label_vector

    model.train()
    model.save(model_filename, include_training_data=True, overwrite=True)

else:
    model.load(model_filename)


# Calculate abundances for all the other stars in the sample!
#testing_set_labels = model.pre
# Plot how we are compared to APOGEE