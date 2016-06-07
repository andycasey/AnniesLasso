
"""
Perform the 17-label training for a few Lambda parameters for the red giant
branch sample that we normalized ourselves.

"""

import cPickle as pickle
import numpy as np
import os
from astropy.table import Table

import AnniesLasso as tc

np.random.seed(123) # For reproducibility.

#base_10_Lambda, scale_factor = sys.argv[1], sys.argv[2]
#base_10_Lambda, scale_factor = 1, 0.5

print("USING BASE 10 {base_10_Lambda} AND SCALE FACTOR {scale_factor}".format(
    scale_factor=scale_factor, base_10_Lambda=base_10_Lambda))

# Data.
PATH, CATALOG, FILE_FORMAT = ("/Users/arc/research/apogee/", "apogee-rg.fits",
    "apogee-rg-custom-normalization-{}.memmap")

# Load the data.
labelled_set = Table.read(os.path.join(PATH, CATALOG))
dispersion = np.memmap(os.path.join(PATH, FILE_FORMAT).format("dispersion"),
    mode="r", dtype=float)
normalized_flux = np.memmap(
    os.path.join(PATH, FILE_FORMAT).format("flux"),
    mode="r", dtype=float).reshape((len(labelled_set), -1))
normalized_ivar = np.memmap(
    os.path.join(PATH, FILE_FORMAT).format("ivar"),
    mode="r", dtype=float).reshape(normalized_flux.shape)

elements = [label_name for label_name in labelled_set.dtype.names \
    if label_name not in ("PARAM_M_H", "SRC_H") and label_name.endswith("_H")]

# Split up the data into ten random subsets.
q = np.random.randint(0, 10, len(labelled_set)) % 10

validate_set = (q == 0)
train_set = (~validate_set)

# Create a vectorizer for all models.
vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(labelled_set,
    tc.vectorizer.polynomial.terminator(["TEFF", "LOGG", "FE_H"], 2),
    scale_factor=scale_factor)

# Create a model and train it on 9/10ths the subset.
model = tc.L1RegularizedCannonModel(labelled_set[train_set],
    normalized_flux[train_set], normalized_ivar[train_set],
    dispersion=dispersion)
model.vectorizer = vectorizer
model.s2 = 0.0
model.regularization = 10**base_10_Lambda

# Train it.
model.train(fixed_scatter=True, use_neighbouring_pixel_theta=True)

# Save it.
filename = "apogee-rg-custom-{0:.2f}-{1:.2e}.model".format(
    base_10_Lambda, scale_factor)
with open(filename, "wb") as fp:
    pickle.dump((model.theta, model.s2, model.regularization, model._metadata), fp, -1)

# Fit the remaining set of normalized spectra (just as a check: we will need to
# do this for the individual stuff too.)
inferred_labels = model.fit(normalized_flux[validate_set], normalized_ivar[validate_set])
expected_labels = np.vstack([labelled_set[label_name][validate_set] \
    for label_name in model.vectorizer.label_names]).T

for i, label_name in enumerate(model.vectorizer.label_names):
    
    difference = expected_labels[:, i] - inferred_labels[:, i]
    print(i, label_name, np.median(difference), np.std(difference))


