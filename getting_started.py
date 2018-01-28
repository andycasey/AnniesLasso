
"""
Getting started with The Cannon and APOGEE
"""

import os
import numpy as np
from astropy.table import Table
import AnniesLasso as tc


# Load in the data.
PATH, CATALOG, FILE_FORMAT = ("/Users/arc/research/apogee/", "apogee-rg.fits",
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


# The labelled set includes ~14000 stars. Let's chose a random ~1,400 for the
# training and validation sets.
np.random.seed(888) # For reproducibility.
q = np.random.randint(0, 10, len(labelled_set)) % 10
validate_set = (q == 0)
train_set = (q == 1)


# Create a Cannon model in parallel using all available threads
model = tc.L1RegularizedCannonModel(labelled_set[train_set],
    normalized_flux[train_set], normalized_ivar[train_set],
    dispersion=dispersion, threads=-1)

# No regularization.
model.regularization = 0

# Specify the vectorizer.
model.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(
    labelled_set[train_set],
    tc.vectorizer.polynomial.terminator(["TEFF", "LOGG", "FE_H"], 2))

print("Vectorizer terms: {0}".format(
    " + ".join(model.vectorizer.get_human_readable_label_vector())))

# Train the model.
model.train()

# Let's set the scatter for each pixel to ensure the mean chi-squared value is
# 1 for the training set, then re-train.
model._set_s2_by_hogg_heuristic()
model.train()


# Use the model to fit the stars in the validation set.
validation_set_labels = model.fit(
    normalized_flux[validate_set], normalized_ivar[validate_set])

for i, label_name in enumerate(model.vectorizer.label_names):
    fig, ax = plt.subplots()
    x = labelled_set[label_name][validate_set]
    y = validation_set_labels[:, i]
    abs_diff = np.abs(y - x)
    ax.scatter(x, y, facecolor="k")

    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    ax.set_xlim(limits.min(), limits.max())
    ax.set_ylim(limits.min(), limits.max())

    ax.set_title("{0}: {1:.2f}".format(label_name, np.mean(abs_diff)))

    print("{0}: {1:.2f}".format(label_name, np.mean(abs_diff)))
