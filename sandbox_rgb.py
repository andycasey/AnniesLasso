
"""
Perform the 17-label training for a few Lambda parameters for the red giant
branch sample.
"""

import cPickle as pickle
import numpy as np
import os
from sys import maxsize
from astropy.table import Table

import AnniesLasso as tc

np.random.seed(123) # For reproducibility.

# Some "configurable" options..
threads = 1
mod = 10

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
q = np.random.randint(0, maxsize, len(labelled_set)) % mod

validate_set = (q == 0)
test_set = (q == (mod - 1))
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

model_filename = "apogee-rg-standard-cannon.model"
if not os.path.exists(model_filename):
    standard_cannon.train()
    standard_cannon.save(model_filename, overwrite=True)
else:
    standard_cannon.load(model_filename)

# Predict labels for the last 1/10th (test) set and compare them to ASCAP.
"""
aspcap_labels = np.array([labelled_set[label_name][test_set] \
    for label_name in vectorizer.label_names]).T
standard_cannon_predicted_labels = standard_cannon.fit(
    normalized_flux[test_set], normalized_ivar[test_set])
"""

regularized_cannon = tc.L1RegularizedCannonModel(labelled_set[train_set],
    normalized_flux[train_set], normalized_ivar[train_set],
    dispersion=dispersion, threads=threads)
regularized_cannon.vectorizer = vectorizer
regularized_cannon.regularization = 10**4

model_filename = "apogee-rg-regularized-cannon.model"
if not os.path.exists(model_filename):
    regularized_cannon.train(initial_theta=True)
    regularized_cannon.save(model_filename, overwrite=True)
else:
    regularized_cannon.load(model_filename)


raise a


# For ~50 pixels, try some Lambda values.
# Recommended pixels from Melissa, in vacuum.
wavelengths = [
    16795.77313988085, # --> continuum
    15339.0, # --> Teff sensitivity
    15720,   # --> Teff sensitivity
    15770,   # --> logg sensitivity
    16811.5, # --> Logg sensitivity
    15221.5, # --> [Fe/H] sensitivity
    16369,   # --> [alpha/Fe] sensitivity
]

# These lines were all given in air, so I have converted them to vacuum.
# Three Fe I lines from Smith et al. (2013)
# Air: 15490.339, 15648.510, 15964.867
# Vac: [15494.571901901722, 15652.785921456885, 15969.22897071544]
wavelengths.extend(
    [15494.571901901722, 15652.785921456885, 15969.22897071544])

# Two Mg I lines from Smith et al. (2013)
# Air: 15765.8, 15879.5
# Vac: [15770.107823467057, 15883.838750072413]
wavelengths.extend(
    [15770.107823467057, 15883.838750072413])

# Two Al I lines from Smith et al. (2013)
# Air: 16718.957, 16763.359
# Vac: [16723.524113765838, 16767.938194147067]
wavelengths.extend(
    [16723.524113765838, 16767.938194147067])

# Two Si II lines from Smith et al. (2013)
# Air: 15960.063, 16060.009
# Vac: [15964.42366396639, 16064.396850911658]
wavelengths.extend(
    [15964.42366396639, 16064.396850911658])

# Two Ca I lines from Smith et al. (2013)
# Air: 16150.763, 16155.236
# Vac: [16155.17553813612, 16159.649754913306]
wavelengths.extend(
    [16155.17553813612, 16159.649754913306])

# Two Cr I lines from Smith et al. (2013)
# Air: 15680.063, 15860.214
# Vac: [15684.347503529489, 15864.547504173868]
wavelengths.extend(
    [15684.347503529489, 15864.547504173868])

# One Co I line from Smith et al. (2013)
# Air: 16757.7
# Vac: 16762.27765450469
wavelengths.extend(
    [16762.27765450469])

# One V I line from Smith et al. (2013) 
# Air: 15924.
# Vac: 15928.350854428922
wavelengths.extend(
    [15928.350854428922])
    
# Two Ni I lines from Smith et al. (2013)
# Air: 16589.295, 16673.711
# Vac: [16593.826837590106, 16678.26580389432]
wavelengths.extend(
    [16593.826837590106, 16678.26580389432])

# Two K I line from Smith et al. (2013)
# Air: 15163.067,15168.376
# Vac: [15167.21089680259, 15172.521340566429]
wavelengths.extend(
    [15167.21089680259, 15172.521340566429])

# Two Mn I lines from Smith et al. (2013)
# Air: 15217., 15262.
# Vac:  [15221.158563809242, 15266.17080169663]
wavelengths.extend(
     [15221.158563809242, 15266.17080169663])


# Set the model up for validation.
Lambdas = 10**np.arange(1, 8.2, 0.2)
regularized_cannon.s2 = 0.0
pixel_mask = np.searchsorted(model.dispersion, wavelengths)

output_filename = "apogee-rg-validation-all.pkl"
if not os.path.exists(output_filename):
    regularizations, chi_sq, log_det, all_models = \
        regularized_cannon.validate_regularization(
            fixed_scatter=True, Lambdas=Lambdas, pixel_mask=pixel_mask, 
            model_filename_format="apogee-rg-validation-{}.pkl", overwrite=True,
            include_training_data=True)

    with open(output_filename, "wb") as fp:
        pickle.dump((regularizations, chi_sq, log_det), fp, -1)

else:
    with open(output_filename, "rb") as fp:
        regularizations, chi_sq, log_det = pickle.load(fp)


# Do a full training


# Plot stuff?

raise a

# Choose a single Lambda value for all pixels.


# Create a 10^5 regularization model and train it on 7/10ths the subset.
regularized_cannon.train()

regularized_cannon.save("apogee-rg-regularized-cannon.model", overwrite=True)

# Predict labels for the last 1/10th set and compare them to ASPCAP.
regularized_cannon_predicted_labels = regularized_cannon.fit(
    normalized_flux[test_set], normalized_ivar[test_set])


# Are we doing as good, or better than ASPCAP?


# How many terms did we remove? --> How has the sparsity changed?

